#pragma once
// bout_filter_core.h — native walking-bout filtering (no Python, no ImGui deps).
//
// Ported 1:1 from Rob Johnson's scripts/bout_segmentation.py
// (origin/rob_ui_overhaul): frame-level filters (hard per-keypoint confidence
// floor, upright posture, floor-Z, Y/X arena walls) -> contiguous-region
// detection with gap bridging -> immobility splitting -> walking validation
// (swing-cycle count + travelled distance). The swing-cycle detector
// reimplements scipy.signal.find_peaks (local maxima -> distance -> prominence
// -> width) so cycle counts match the original script.
//
// Operates on DENSE per-frame arrays (length = total_video_frames, NaN for
// frames absent from the sparse .rpred store). Coordinates are in mm (the
// caller divides raw store coords by profile scale); confidence is unscaled.

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace boutfilter {

struct Params {
    bool  confidence_enabled = true;
    float confidence_threshold = 0.80f;   // hard floor: EVERY keypoint >= this
    int   confidence_gap_bridge = 15;
    bool  upright_enabled = true;
    bool  floor_z_enabled = true;
    float floor_z_threshold = 0.40f;
    float y_wall_min = 0.0f;
    float y_wall_max = 5.4f;
    float x_wall_margin = 0.0f;
    int   immobility_max_frames = 25;
    float immobility_speed_threshold = 5.0f;
    int   min_bout_frames = 100;
    int   max_gap_bridge = 100;
    int   min_walking_cycles = 2;
    float min_distance_mm = 5.0f;
    int   max_swing_duration = 35;
    float swing_prominence = 0.05f;
};

// Per-keypoint dense trajectory in mm (NaN where the frame is absent).
struct KpTrack {
    std::vector<float> x, y, z;
};

struct Inputs {
    int total_frames = 0;
    int fps = 800;
    float arena_x_mm = 23.5f;
    std::vector<float> min_conf;         // per-frame min confidence over ALL keypoints
    std::vector<KpTrack> leg_tips;       // 6 leg tips
    std::vector<char> leg_is_x_wall;     // parallel to leg_tips: use for X-wall test
    KpTrack body_ref;                    // scutellum / thorax
};

// Provenance of a displayed bout / a forced status. Manual edits are anchored
// to frame ranges (not bout indices) so they survive a wholesale recompute.
enum class EditKind     : uint8_t { None = 0, Merge = 1, Adjust = 2, Manual = 3 };
enum class ForcedStatus : uint8_t { None = 0, Accept = 1, Reject = 2 };

struct ResultBout {
    int start = 0, end = 0, n_frames = 0;
    double duration_s = 0;
    int min_cycles = 0;
    bool accepted = false;
    std::string reason;
    double total_distance_mm = NAN, net_displacement_mm = NAN;
    double mean_speed_mm_s = NAN, max_speed_mm_s = NAN, body_z_mean = NAN;
    // Manual-edit provenance (None/0 => pure auto-detected).
    EditKind edit_kind = EditKind::None;
    bool     status_overridden = false;   // accept/reject was manually forced
    uint64_t edit_id = 0;                 // links to the overlay entry (0 = none)
};

struct Span { int start, end; };  // inclusive

struct Result {
    std::vector<ResultBout> bouts;
    std::vector<Span> conf_spans, upright_spans, floor_spans, ywall_spans, xwall_spans;
    int n_frames = 0, n_valid_frames = 0, n_candidates = 0;
    int n_accepted = 0, n_rejected = 0;
    double pct_conf = 0, pct_upright = 0, pct_floor_ok = 0,
           pct_ywall_ok = 0, pct_xwall_ok = 0, pct_all = 0;
    std::vector<std::string> edit_warnings;   // stale/orphaned manual edits
};

// ── manual-edit overlay ──────────────────────────────────────────────────────
// A merge and a boundary-adjust both reduce to a manual bout with an explicit,
// user-authoritative [start,end] whose metrics are recomputed via evaluate_bout.
struct ManualBout {
    uint64_t     id = 0;
    int          start = 0, end = 0;      // inclusive, authoritative
    ForcedStatus forced = ForcedStatus::None;
    EditKind     kind = EditKind::Manual; // Merge or Adjust (provenance for UI/CSV)
};

// An accept/reject flip on an otherwise auto-detected bout, anchored to the
// auto bout's frame range at the time the override was made.
struct StatusOverride {
    uint64_t     id = 0;
    Span         anchor{0, 0};
    ForcedStatus status = ForcedStatus::Accept;
};

struct BoutEdits {
    int schema_version = 1;
    uint64_t next_id = 1;                  // monotonic id source (persisted)
    std::vector<ManualBout>     manual_bouts;
    std::vector<StatusOverride> overrides;
    bool empty() const { return manual_bouts.empty() && overrides.empty(); }
};

// ── low-level numeric helpers (mirror numpy/scipy) ───────────────────────────

inline std::vector<char> bridge_gaps(const std::vector<char> &mask, int max_gap) {
    if (max_gap <= 0) return mask;
    std::vector<char> out = mask;
    bool in_gap = false;
    int gap_start = 0;
    const int n = (int)out.size();
    for (int i = 0; i < n; ++i) {
        if (!out[i]) {
            if (!in_gap) { gap_start = i; in_gap = true; }
        } else {
            if (in_gap) {
                if ((i - gap_start) <= max_gap && gap_start > 0)
                    for (int k = gap_start; k < i; ++k) out[k] = 1;
                in_gap = false;
            }
        }
    }
    return out;
}

// np.interp-style linear fill of NaNs (endpoints clamp to nearest valid).
// Returns false if the signal is entirely NaN.
inline bool interp_nans(std::vector<float> &z) {
    const int n = (int)z.size();
    std::vector<int> vi;
    vi.reserve(n);
    for (int i = 0; i < n; ++i) if (!std::isnan(z[i])) vi.push_back(i);
    if (vi.empty()) return false;
    for (int i = 0; i < n; ++i) {
        if (!std::isnan(z[i])) continue;
        if (i <= vi.front()) { z[i] = z[vi.front()]; continue; }
        if (i >= vi.back())  { z[i] = z[vi.back()];  continue; }
        // find surrounding valid indices (binary search)
        int lo = 0, hi = (int)vi.size() - 1;
        while (hi - lo > 1) { int m = (lo + hi) / 2; if (vi[m] <= i) lo = m; else hi = m; }
        int a = vi[lo], b = vi[hi];
        float t = (float)(i - a) / (float)(b - a);
        z[i] = z[a] + t * (z[b] - z[a]);
    }
    return true;
}

// scipy uniform_filter1d(size, mode='reflect', origin=0).
inline std::vector<float> uniform_filter1d(const std::vector<float> &x, int size) {
    const int n = (int)x.size();
    std::vector<float> out(n);
    if (n == 0) return out;
    int r = size / 2;              // symmetric radius for odd size
    int lo = -r, hi = size - 1 - r;
    for (int i = 0; i < n; ++i) {
        double s = 0;
        for (int k = lo; k <= hi; ++k) {
            int j = i + k;
            // reflect about edge of last pixel: (d c b a | a b c d | d c b a)
            while (j < 0 || j >= n) {
                if (j < 0) j = -j - 1;
                if (j >= n) j = 2 * n - j - 1;
            }
            s += x[j];
        }
        out[i] = (float)(s / size);
    }
    return out;
}

// scipy _local_maxima_1d.
inline std::vector<int> local_maxima_1d(const std::vector<float> &x) {
    std::vector<int> mids;
    const int n = (int)x.size();
    int i = 1, i_max = n - 1;
    while (i < i_max) {
        if (x[i - 1] < x[i]) {
            int i_ahead = i + 1;
            while (i_ahead < i_max && x[i_ahead] == x[i]) ++i_ahead;
            if (x[i_ahead] < x[i]) {
                int left = i, right = i_ahead - 1;
                mids.push_back((left + right) / 2);
                i = i_ahead;
            }
        }
        ++i;
    }
    return mids;
}

// scipy _select_by_peak_distance. Keeps highest peaks, removing any within
// `distance` of a kept higher peak. Returns filtered peak indices.
inline std::vector<int> select_by_distance(const std::vector<int> &peaks,
                                           const std::vector<float> &x,
                                           double distance) {
    const int n = (int)peaks.size();
    std::vector<char> keep(n, 1);
    int dist = (int)std::ceil(distance);
    // argsort by peak height ascending
    std::vector<int> order(n);
    for (int i = 0; i < n; ++i) order[i] = i;
    std::sort(order.begin(), order.end(),
              [&](int a, int b) { return x[peaks[a]] < x[peaks[b]]; });
    for (int idx = n - 1; idx >= 0; --idx) {
        int j = order[idx];
        if (!keep[j]) continue;
        int k = j - 1;
        while (k >= 0 && peaks[j] - peaks[k] < dist) { keep[k] = 0; --k; }
        k = j + 1;
        while (k < n && peaks[k] - peaks[j] < dist) { keep[k] = 0; ++k; }
    }
    std::vector<int> out;
    for (int i = 0; i < n; ++i) if (keep[i]) out.push_back(peaks[i]);
    return out;
}

// scipy _peak_prominences (wlen = full signal).
inline void peak_prominences(const std::vector<float> &x,
                             const std::vector<int> &peaks,
                             std::vector<float> &prom,
                             std::vector<int> &lbase,
                             std::vector<int> &rbase) {
    const int n = (int)x.size();
    const int np_ = (int)peaks.size();
    prom.assign(np_, 0); lbase.assign(np_, 0); rbase.assign(np_, 0);
    for (int p = 0; p < np_; ++p) {
        int peak = peaks[p];
        int i_min = 0, i_max = n - 1;
        // left base
        lbase[p] = peak;
        float left_min = x[peak];
        int i = peak;
        while (i_min <= i && x[i] <= x[peak]) {
            if (x[i] < left_min) { left_min = x[i]; lbase[p] = i; }
            --i;
        }
        // right base
        rbase[p] = peak;
        float right_min = x[peak];
        i = peak;
        while (i <= i_max && x[i] <= x[peak]) {
            if (x[i] < right_min) { right_min = x[i]; rbase[p] = i; }
            ++i;
        }
        prom[p] = x[peak] - std::max(left_min, right_min);
    }
}

// scipy _peak_widths at rel_height=0.5.
inline std::vector<float> peak_widths(const std::vector<float> &x,
                                      const std::vector<int> &peaks,
                                      const std::vector<float> &prom,
                                      const std::vector<int> &lbase,
                                      const std::vector<int> &rbase,
                                      float rel_height = 0.5f) {
    const int np_ = (int)peaks.size();
    std::vector<float> widths(np_, 0);
    for (int p = 0; p < np_; ++p) {
        int peak = peaks[p];
        int i_min = lbase[p], i_max = rbase[p];
        float height = x[peak] - prom[p] * rel_height;
        // left intersection
        int i = peak;
        while (i_min < i && height < x[i]) --i;
        float left_ip = (float)i;
        if (x[i] < height) left_ip = (float)i + (height - x[i]) / (x[i + 1] - x[i]);
        // right intersection
        i = peak;
        while (i < i_max && height < x[i]) ++i;
        float right_ip = (float)i;
        if (x[i] < height) right_ip = (float)i - (height - x[i]) / (x[i - 1] - x[i]);
        widths[p] = right_ip - left_ip;
    }
    return widths;
}

// count_swing_phases: scipy find_peaks(prominence, distance=8, width=(1,max)).
inline int count_swing_phases(std::vector<float> z, const Params &p) {
    if (!interp_nans(z)) return 0;
    std::vector<float> zs = uniform_filter1d(z, 5);
    std::vector<int> peaks = local_maxima_1d(zs);
    if (peaks.empty()) return 0;
    peaks = select_by_distance(peaks, zs, 8.0);
    if (peaks.empty()) return 0;
    std::vector<float> prom; std::vector<int> lb, rb;
    peak_prominences(zs, peaks, prom, lb, rb);
    // prominence filter
    std::vector<int> pk2; std::vector<float> pr2; std::vector<int> lb2, rb2;
    for (int i = 0; i < (int)peaks.size(); ++i) {
        if (prom[i] >= p.swing_prominence) {
            pk2.push_back(peaks[i]); pr2.push_back(prom[i]);
            lb2.push_back(lb[i]); rb2.push_back(rb[i]);
        }
    }
    if (pk2.empty()) return 0;
    std::vector<float> w = peak_widths(zs, pk2, pr2, lb2, rb2, 0.5f);
    int count = 0;
    for (float wid : w)
        if (wid >= 1.0f && wid <= (float)p.max_swing_duration) ++count;
    return count;
}

// ── bout detection ───────────────────────────────────────────────────────────

inline std::vector<Span> find_bouts(const std::vector<char> &valid,
                                    int min_frames, int max_gap) {
    std::vector<char> bridged = bridge_gaps(valid, max_gap);
    std::vector<Span> bouts;
    bool in_bout = false; int start = 0;
    const int n = (int)bridged.size();
    for (int i = 0; i < n; ++i) {
        if (bridged[i] && !in_bout) { start = i; in_bout = true; }
        else if (!bridged[i] && in_bout) {
            if (i - start >= min_frames) bouts.push_back({start, i - 1});
            in_bout = false;
        }
    }
    if (in_bout && n - start >= min_frames) bouts.push_back({start, n - 1});
    return bouts;
}

inline std::vector<float> compute_speed(const KpTrack &b, int start, int end, int fps) {
    int len = end - start + 1;
    std::vector<float> spd(len, 0);
    if (len <= 0) return spd;
    for (int i = 1; i < len; ++i) {
        float dx = b.x[start + i] - b.x[start + i - 1];
        float dy = b.y[start + i] - b.y[start + i - 1];
        spd[i] = std::sqrt(dx * dx + dy * dy) * fps;
    }
    spd[0] = len > 1 ? spd[1] : 0;  // Python prepends spd[0] (first diff); match shape
    // Python: np.concatenate([[spd[0]], spd]) where spd = diffs -> first element
    // duplicates the first diff. spd[1] here is that first diff. For len==1, 0.
    return spd;
}

// Re-splits each candidate bout wherever a floor-Z, Y-wall, X-wall, or
// immobility violation occurs — mirrors Python's split_bouts_at_violations,
// which combines all four ("split_bout" mode) into one violation mask before
// re-scanning for >= min_bout_frames sub-runs. Without this, gap-bridging in
// find_bouts() can merge a wall/floor touch into the middle of a candidate
// bout instead of cutting it out.
inline std::vector<Span> split_at_violations(const std::vector<Span> &bouts,
                                             const KpTrack &body,
                                             const std::vector<char> &floor_viol,
                                             const std::vector<char> &ywall_viol,
                                             const std::vector<char> &xwall_viol,
                                             const Params &p, int fps) {
    std::vector<Span> result;
    for (const auto &bt : bouts) {
        int start = bt.start, end = bt.end;
        int len = end - start + 1;
        std::vector<char> viol(len, 0);
        for (int i = 0; i < len; ++i)
            viol[i] = floor_viol[start + i] || ywall_viol[start + i] || xwall_viol[start + i];

        std::vector<float> spd = compute_speed(body, start, end, fps);
        int run_len = 0, run_start = 0;
        for (int i = 0; i < len; ++i) {
            bool stat = spd[i] < p.immobility_speed_threshold;
            if (stat) { if (run_len == 0) run_start = i; ++run_len; }
            else {
                if (run_len > p.immobility_max_frames)
                    for (int k = run_start; k < run_start + run_len; ++k) viol[k] = 1;
                run_len = 0;
            }
        }
        if (run_len > p.immobility_max_frames)
            for (int k = run_start; k < run_start + run_len; ++k) viol[k] = 1;

        bool in_valid = false; int sub_start = 0;
        for (int i = 0; i < len; ++i) {
            bool ok = !viol[i];
            if (ok && !in_valid) { sub_start = i; in_valid = true; }
            else if (!ok && in_valid) {
                if (i - sub_start >= p.min_bout_frames)
                    result.push_back({start + sub_start, start + i - 1});
                in_valid = false;
            }
        }
        if (in_valid && (len - sub_start) >= p.min_bout_frames)
            result.push_back({start + sub_start, end});
    }
    return result;
}

// ── run-length encode a boolean (violation=true) mask ────────────────────────
inline std::vector<Span> rle_true(const std::vector<char> &m) {
    std::vector<Span> spans;
    const int n = (int)m.size();
    int i = 0;
    while (i < n) {
        if (m[i]) {
            int j = i;
            while (j + 1 < n && m[j + 1]) ++j;
            spans.push_back({i, j});
            i = j + 1;
        } else ++i;
    }
    return spans;
}

// ── evaluate one candidate/manual bout ───────────────────────────────────────
// Computes metrics + the accept/reject decision for the inclusive frame range
// [start,end]. Extracted from compute()'s per-candidate loop so manual
// merge/adjust edits recompute metrics with the exact same logic. Unlike
// compute()'s candidates (always in bounds), manual ranges are user-supplied,
// so this clamps to the stored frame range and rejects an empty/invalid span.
inline ResultBout evaluate_bout(const Inputs &in, const Params &p, int start, int end) {
    ResultBout b;
    const int n = in.total_frames;
    if (n <= 0) { b.start = start; b.end = end; b.reason = "invalid range"; return b; }
    if (start < 0) start = 0;
    if (end > n - 1) end = n - 1;
    if (start > end) { b.start = start; b.end = end; b.reason = "invalid range"; return b; }

    b.start = start; b.end = end;
    b.n_frames = end - start + 1;
    b.duration_s = (double)b.n_frames / in.fps;

    int min_cycles = INT32_MAX;
    for (const auto &leg : in.leg_tips) {
        std::vector<float> seg(leg.z.begin() + start, leg.z.begin() + end + 1);
        min_cycles = std::min(min_cycles, count_swing_phases(seg, p));
    }
    if (min_cycles == INT32_MAX) min_cycles = 0;
    b.min_cycles = min_cycles;
    if (min_cycles < p.min_walking_cycles) {
        b.accepted = false;
        b.reason = "too few walking cycles (" + std::to_string(min_cycles) + ")";
        return b;
    }

    // travelled distance (finite frames only)
    double total = 0, netx = 0, nety = 0;
    int nfin = 0; float fx0 = 0, fy0 = 0, fxl = 0, fyl = 0;
    float px = NAN, py = NAN;
    for (int i = start; i <= end; ++i) {
        float x = in.body_ref.x[i], y = in.body_ref.y[i];
        if (std::isnan(x) || std::isnan(y)) continue;
        if (nfin == 0) { fx0 = x; fy0 = y; }
        else { float dx = x - px, dy = y - py; total += std::sqrt(dx*dx + dy*dy); }
        fxl = x; fyl = y; px = x; py = y; ++nfin;
    }
    if (nfin < 2) {
        b.accepted = false; b.reason = "insufficient valid frames";
        return b;
    }
    netx = fxl - fx0; nety = fyl - fy0;
    b.total_distance_mm = total;
    b.net_displacement_mm = std::sqrt(netx*netx + nety*nety);
    if (total < p.min_distance_mm) {
        b.accepted = false;
        char buf[64]; snprintf(buf, sizeof(buf), "too short (%.1f mm)", total);
        b.reason = buf;
        return b;
    }

    std::vector<float> spd = compute_speed(in.body_ref, start, end, in.fps);
    double smean = 0, smax = 0; int sc = 0;
    for (float s : spd) { if (!std::isnan(s)) { smean += s; smax = std::max(smax, (double)s); ++sc; } }
    double zmean = 0; int zc = 0;
    for (int i = start; i <= end; ++i) { float z = in.body_ref.z[i]; if (!std::isnan(z)) { zmean += z; ++zc; } }
    b.mean_speed_mm_s = sc ? smean / sc : 0;
    b.max_speed_mm_s = smax;
    b.body_z_mean = zc ? zmean / zc : NAN;
    b.accepted = true;
    return b;
}

// ── full pipeline ─────────────────────────────────────────────────────────────

inline Result compute(const Inputs &in, const Params &p) {
    Result R;
    const int n = in.total_frames;
    R.n_frames = n;
    if (n <= 0) return R;

    std::vector<char> conf_mask(n, 1), upright_mask(n, 1),
                      floor_viol(n, 0), ywall_viol(n, 0), xwall_viol(n, 0);

    for (int i = 0; i < n; ++i) {
        // confidence floor (NaN fails)
        if (p.confidence_enabled) {
            float mc = in.min_conf[i];
            conf_mask[i] = (!std::isnan(mc) && mc >= p.confidence_threshold) ? 1 : 0;
        }
        // upright: body_ref z above every leg tip z (NaN comparison -> false)
        if (p.upright_enabled) {
            bool up = true;
            float bz = in.body_ref.z[i];
            for (const auto &leg : in.leg_tips)
                up = up && (bz > leg.z[i]);   // NaN>NaN == false
            upright_mask[i] = up ? 1 : 0;
        }
        // floor-Z: any leg below floor
        if (p.floor_z_enabled) {
            bool v = false;
            for (const auto &leg : in.leg_tips) v = v || (leg.z[i] < p.floor_z_threshold);
            floor_viol[i] = v ? 1 : 0;
        }
        // Y walls
        {
            bool v = false;
            for (const auto &leg : in.leg_tips)
                v = v || (leg.y[i] >= p.y_wall_max) || (leg.y[i] <= p.y_wall_min);
            ywall_viol[i] = v ? 1 : 0;
        }
        // X walls (only x-wall legs)
        {
            bool v = false;
            for (size_t l = 0; l < in.leg_tips.size(); ++l) {
                if (!in.leg_is_x_wall[l]) continue;
                float x = in.leg_tips[l].x[i];
                v = v || (x >= in.arena_x_mm - p.x_wall_margin) || (x <= p.x_wall_margin);
            }
            xwall_viol[i] = v ? 1 : 0;
        }
    }

    if (p.confidence_enabled)
        conf_mask = bridge_gaps(conf_mask, p.confidence_gap_bridge);

    std::vector<char> valid(n, 0);
    for (int i = 0; i < n; ++i)
        valid[i] = (conf_mask[i] && upright_mask[i] && !floor_viol[i] &&
                    !ywall_viol[i] && !xwall_viol[i]) ? 1 : 0;

    std::vector<Span> candidates = find_bouts(valid, p.min_bout_frames, p.max_gap_bridge);
    candidates = split_at_violations(candidates, in.body_ref, floor_viol, ywall_viol, xwall_viol, p, in.fps);
    R.n_candidates = (int)candidates.size();

    for (const auto &c : candidates)
        R.bouts.push_back(evaluate_bout(in, p, c.start, c.end));

    for (auto &b : R.bouts) (b.accepted ? R.n_accepted : R.n_rejected)++;

    R.conf_spans    = rle_true([&]{ std::vector<char> m(n); for (int i=0;i<n;++i) m[i]=!conf_mask[i]; return m; }());
    R.upright_spans = rle_true([&]{ std::vector<char> m(n); for (int i=0;i<n;++i) m[i]=!upright_mask[i]; return m; }());
    R.floor_spans   = rle_true(floor_viol);
    R.ywall_spans   = rle_true(ywall_viol);
    R.xwall_spans   = rle_true(xwall_viol);

    auto count_true = [&](const std::vector<char>&m){ long c=0; for(char v:m) c+=v?1:0; return c; };
    auto pct = [&](long c){ return n ? 100.0 * c / n : 0.0; };
    long nvalid = count_true(valid);
    R.n_valid_frames = (int)nvalid;
    R.pct_conf = pct(count_true(conf_mask));
    R.pct_upright = pct(count_true(upright_mask));
    R.pct_floor_ok = pct(n - count_true(floor_viol));
    R.pct_ywall_ok = pct(n - count_true(ywall_viol));
    R.pct_xwall_ok = pct(n - count_true(xwall_viol));
    R.pct_all = pct(nvalid);
    return R;
}

// ── manual-edit overlay replay ────────────────────────────────────────────────
// Transforms a fresh compute() result (all rows edit_kind==None) into the
// curated list by replaying the manual overlay. Deterministic: depends only on
// (R, edits, in, p) and fixed tie-break rules, so the curated list is identical
// on every recompute. Only overrides can go stale (they decorate auto bouts);
// a manual bout that overlaps no auto bout is authoritative and persists.
inline void apply_bout_edits(Result &R, const BoutEdits &edits,
                             const Inputs &in, const Params &p) {
    const int total = in.total_frames;
    auto stronger = [](ForcedStatus a, ForcedStatus b) {
        // Reject > Accept > None
        auto rank = [](ForcedStatus s) {
            return s == ForcedStatus::Reject ? 2 : s == ForcedStatus::Accept ? 1 : 0;
        };
        return rank(a) >= rank(b) ? a : b;
    };

    // 1. Coalesce manual bouts: drop fully-invalid/out-of-range (orphans),
    //    clamp partial, sort by start, union any that overlap or touch.
    std::vector<ManualBout> mb;
    for (const auto &m : edits.manual_bouts) {
        if (m.start > m.end || m.end < 0 || m.start > total - 1 || total <= 0) {
            R.edit_warnings.push_back(
                "manual bout " + std::to_string(m.start) + "-" + std::to_string(m.end) +
                " is outside the stored frame range");
            continue;
        }
        ManualBout c = m;
        if (c.start < 0) c.start = 0;
        if (c.end > total - 1) c.end = total - 1;
        mb.push_back(c);
    }
    std::sort(mb.begin(), mb.end(),
              [](const ManualBout &a, const ManualBout &b) { return a.start < b.start; });
    std::vector<ManualBout> merged;
    for (const auto &c : mb) {
        if (!merged.empty() && c.start <= merged.back().end + 1) {
            ManualBout &top = merged.back();
            top.end = std::max(top.end, c.end);
            top.forced = stronger(top.forced, c.forced);
            top.kind = EditKind::Merge;   // a union is a merge
        } else {
            merged.push_back(c);
        }
    }

    // 2. Suppress auto bouts overlapping any coalesced manual range (whole
    //    bout consumed — no partial truncation).
    std::vector<ResultBout> kept;
    for (const auto &a : R.bouts) {
        bool consumed = false;
        for (const auto &M : merged)
            if (a.start <= M.end && a.end >= M.start) { consumed = true; break; }
        if (!consumed) kept.push_back(a);
    }
    R.bouts.swap(kept);

    // 3. Materialize each coalesced manual bout with recomputed metrics.
    for (const auto &M : merged) {
        ResultBout b = evaluate_bout(in, p, M.start, M.end);
        b.edit_kind = M.kind;
        b.edit_id = M.id;
        if (M.forced == ForcedStatus::Accept) {
            if (!b.accepted) b.reason = "manually accepted";
            b.accepted = true;
            b.status_overridden = true;
        } else if (M.forced == ForcedStatus::Reject) {
            b.accepted = false;
            b.reason = "manually rejected";
            b.status_overridden = true;
        }
        R.bouts.push_back(b);
    }

    // 4. Apply status overrides to the best-matching surviving auto row
    //    (IoU-style overlap >= 0.5). No match => stale.
    for (const auto &O : edits.overrides) {
        int best = -1; double best_iou = 0.0;
        int anchor_len = O.anchor.end - O.anchor.start + 1;
        for (size_t i = 0; i < R.bouts.size(); ++i) {
            ResultBout &b = R.bouts[i];
            if (b.edit_kind != EditKind::None) continue;   // auto rows only
            int ov = std::min(b.end, O.anchor.end) - std::max(b.start, O.anchor.start) + 1;
            if (ov <= 0) continue;
            int denom = std::max(anchor_len, b.n_frames);
            double iou = denom > 0 ? (double)ov / denom : 0.0;
            if (iou > best_iou) { best_iou = iou; best = (int)i; }
        }
        if (best >= 0 && best_iou >= 0.5) {
            ResultBout &b = R.bouts[best];
            b.status_overridden = true;
            b.edit_id = O.id;
            if (O.status == ForcedStatus::Accept) {
                if (!b.accepted) b.reason = "manually accepted";
                b.accepted = true;
            } else {
                b.accepted = false;
                b.reason = "manually rejected";
            }
        } else {
            R.edit_warnings.push_back(
                "status override for frames " + std::to_string(O.anchor.start) + "-" +
                std::to_string(O.anchor.end) + " no longer matches any bout");
        }
    }

    // 5. Re-sort by start and recount (n_candidates stays = auto count;
    //    frame-level percentages describe filters, not bouts, so untouched).
    std::sort(R.bouts.begin(), R.bouts.end(),
              [](const ResultBout &a, const ResultBout &b) { return a.start < b.start; });
    R.n_accepted = R.n_rejected = 0;
    for (const auto &b : R.bouts) (b.accepted ? R.n_accepted : R.n_rejected)++;
}

}  // namespace boutfilter
