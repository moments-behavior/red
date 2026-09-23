#pragma once
// camera_check.h — detect cameras with bad calibration from the 2D the user
// triangulates, plus the helpers for excluding cameras from 3D.
//
// A camera with bad calibration can't be spotted from its yaml alone, but it
// shows up against the others. Plain leave-one-out (triangulate without c,
// project into c) is weak: the bad camera also sits in every *other*
// camera's reference set and inflates their errors too. So the detector is
// drop-one consistency: for each camera k, triangulate from all cameras but
// k and measure how well *those* cameras agree. Dropping the bad camera makes
// the rest agree (~noise); dropping a good one leaves the bad one in and the
// rest disagree. Leave-one-out works well with many cameras, drop-one with
// few, so both are used. Applied greedily, so a second bad camera surfaces
// once the first is set aside. The math lives in gui_keypoints.h
// (analyze_camera_check); this header only holds the samples.
//
// Samples are the *raw* 2D (predictions / manual labels) captured just
// before Triangulate overwrites it with reprojections, for nodes whose 3D
// is not yet triangulated. Keeping observations (not residuals) lets the
// analysis re-run against the current exclusion set.

#include "types.h"
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <map>
#include <string>
#include <vector>

struct CameraCheckStats {
    struct Obs {
        int cam;
        Eigen::Vector2d px;  // image coords (y down), distorted
    };
    // frame -> [node] -> labeled views of that node
    using FrameObs = std::vector<std::vector<Obs>>;
    std::map<u32, FrameObs> per_frame;
    int version = 0;  // bumped on change; lets callers cache the analysis

    static size_t count(const FrameObs &f) {
        size_t n = 0;
        for (const auto &v : f) n += v.size();
        return n;
    }

    // Keep the richest sample per frame: a repeat Triangulate on the same
    // frame only sees the few nodes the user just dragged, which would
    // otherwise replace the full raw-prediction sample.
    void record(u32 frame, FrameObs obs) {
        size_t n = count(obs);
        if (n == 0) return;
        auto it = per_frame.find(frame);
        if (it == per_frame.end() || n >= count(it->second)) {
            per_frame[frame] = std::move(obs);
            version++;
        }
    }

    void clear() {
        per_frame.clear();
        version++;
    }
};

struct CameraCheckResult {
    struct Cam {
        float loo_px = NAN;   // median error vs. the consistent camera set
        float drop_px = NAN;  // median error of the others when this is dropped
        int samples = 0;
        bool suggested = false;
    };
    std::vector<Cam> cams;
    float typical_drop_px = NAN;  // rig consistency with a good camera dropped
    int frames = 0;
};

// Suggestion thresholds for analyze_camera_check().
constexpr float kCamCheckGain = 3.0f;   // x the typical camera's error
constexpr float kCamCheckMinPx = 8.0f;  // absolute floor (= dashboard cut)
constexpr int kCamCheckMinSamples = 10;

// Per-camera exclusion mask, aligned with `camera_names` (= scene cams).
inline std::vector<bool>
camera_exclusion_mask(const std::vector<std::string> &camera_names,
                      const std::vector<std::string> &excluded) {
    std::vector<bool> mask(camera_names.size(), false);
    for (size_t c = 0; c < camera_names.size(); ++c)
        mask[c] = std::find(excluded.begin(), excluded.end(),
                            camera_names[c]) != excluded.end();
    return mask;
}
