#pragma once
// annotation_diagnostics.h — adapter that runs the reprojection diagnostic
// on the labels in an AnnotationMap (an Annotation Project's in-memory
// keypoint data). Pure, read-only: never mutates AnnotationMap.
//
// Handles Y-flip per camera: AnnotationMap stores Y in ImPlot (bottom-left)
// convention, while the calibration math expects Y in top-left (image)
// convention. See annotation_csv.h:16 and gui_keypoints.h:138-139.
//
// Works for both telecentric and perspective cameras — dispatches through
// the same ReprojectionDiagnostics primitive that branches on
// CameraParams.telecentric.

#include "annotation.h"
#include "camera.h"
#include "reprojection_diagnostics.h"
#include <algorithm>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>

namespace AnnotationDiagnostics {

// Label attached to each "virtual point" we send to the primitive.
struct PointLabel {
    u32 frame = 0;
    u32 kp = 0;
};

struct PerKeypointStats {
    u32 kp_id = 0;
    int n_points = 0;   // triangulated (or known-3D) points with this keypoint
    int n_obs = 0;      // residual count contributed by this keypoint
    double mean = 0, std = 0, median = 0, p95 = 0, max = 0, rmse = 0;
};

struct Diagnostics {
    bool success = false;
    std::string error;
    ReprojectionDiagnostics::Diagnostics base;
    // Aligned with base.triangulated / base.residuals.point_id: the (frame, kp)
    // that each virtual point index corresponds to.
    std::vector<PointLabel> point_labels;
    std::vector<PerKeypointStats> per_keypoint;
    // How many (frame, kp) candidates had <2 labeled views and were skipped.
    int n_candidates_skipped_lt2_views = 0;
};

inline double percentile_sorted(const std::vector<double> &s, double q) {
    if (s.empty()) return 0.0;
    double idx = q * (s.size() - 1);
    size_t lo = (size_t)std::floor(idx);
    size_t hi = (size_t)std::ceil(idx);
    double t = idx - lo;
    return s[lo] * (1.0 - t) + s[hi] * t;
}

// Walk an AnnotationMap + compute triangulate-and-reproject diagnostics.
//
//   annotations    — in-memory labels
//   cameras        — one CameraParams per camera (branches on .telecentric)
//   image_heights  — per-camera image height for Y-flip (typically from scene)
//   num_nodes      — skeleton.num_nodes (for iterating keypoints)
inline Diagnostics compute(
    const AnnotationMap &annotations,
    const std::vector<CameraParams> &cameras,
    const std::vector<int> &image_heights,
    int num_nodes) {

    Diagnostics d;

    int M = (int)cameras.size();
    if (M == 0) {
        d.error = "No camera parameters loaded";
        return d;
    }
    if ((int)image_heights.size() != M) {
        d.error = "image_heights size != cameras size";
        return d;
    }
    for (int m = 0; m < M; m++) {
        if (image_heights[m] <= 0) {
            d.error = "image_heights[" + std::to_string(m) +
                      "] not populated (load videos first)";
            return d;
        }
    }
    if (num_nodes <= 0) {
        d.error = "Skeleton has no keypoints";
        return d;
    }
    if (annotations.empty()) {
        d.error = "No labeled frames";
        return d;
    }

    // First pass: collect (frame, kp) candidates that have labels in >= 1 cam.
    // (The primitive itself filters <2-view cases; we track them for reporting.)
    struct Candidate {
        u32 frame;
        u32 kp;
        std::vector<int> cams_labeled;  // camera indices that have this point
    };
    std::vector<Candidate> candidates;
    for (const auto &kv : annotations) {
        u32 frame = kv.first;
        const FrameAnnotation &fa = kv.second;
        for (int kp = 0; kp < num_nodes; kp++) {
            Candidate c;
            c.frame = frame;
            c.kp = (u32)kp;
            for (int m = 0; m < M; m++) {
                if (m >= (int)fa.cameras.size()) continue;
                if (kp >= (int)fa.cameras[m].keypoints.size()) continue;
                const Keypoint2D &k2 = fa.cameras[m].keypoints[kp];
                if (!k2.labeled) continue;
                if (k2.x >= UNLABELED * 0.9 || k2.y >= UNLABELED * 0.9) continue;
                if (!std::isfinite(k2.x) || !std::isfinite(k2.y)) continue;
                c.cams_labeled.push_back(m);
            }
            if (!c.cams_labeled.empty()) {
                if ((int)c.cams_labeled.size() < 2)
                    d.n_candidates_skipped_lt2_views++;
                candidates.push_back(std::move(c));
            }
        }
    }

    if (candidates.empty()) {
        d.error = "No labeled keypoints found";
        return d;
    }

    // Second pass: build the observations[cam][virtual_idx] matrix.
    // Every virtual point has length M; unlabeled cams get the sentinel.
    int N = (int)candidates.size();
    std::vector<std::vector<Eigen::Vector2d>> obs(M);
    for (int m = 0; m < M; m++) {
        obs[m].assign(N, Eigen::Vector2d(
            ReprojectionDiagnostics::UNLABELED_SENTINEL,
            ReprojectionDiagnostics::UNLABELED_SENTINEL));
    }
    d.point_labels.resize(N);

    for (int i = 0; i < N; i++) {
        const Candidate &c = candidates[i];
        d.point_labels[i] = PointLabel{c.frame, c.kp};
        const FrameAnnotation &fa = annotations.at(c.frame);
        for (int m : c.cams_labeled) {
            const Keypoint2D &k2 = fa.cameras[m].keypoints[c.kp];
            // Flip Y: AnnotationMap is bottom-left, math expects top-left.
            obs[m][i] = Eigen::Vector2d(
                k2.x, (double)image_heights[m] - k2.y);
        }
    }

    // Camera display names: use a generic index if CameraParams doesn't carry
    // the serial (CameraParams doesn't have a name field).
    std::vector<std::string> names(M);
    for (int m = 0; m < M; m++) names[m] = "Cam" + std::to_string(m);

    d.base = ReprojectionDiagnostics::compute_diagnostics(obs, cameras, names);
    if (!d.base.success) {
        d.error = d.base.error;
        return d;
    }

    // Per-keypoint aggregation.
    std::vector<std::vector<double>> per_kp_errs(num_nodes);
    std::vector<int> per_kp_n_points(num_nodes, 0);
    for (size_t i = 0; i < d.base.residuals.size(); i++) {
        const auto &r = d.base.residuals[i];
        u32 kp = (r.point_id >= 0 && r.point_id < (int)d.point_labels.size())
            ? d.point_labels[r.point_id].kp : 0;
        if ((int)kp < num_nodes)
            per_kp_errs[kp].push_back(r.error_px);
    }
    // Count triangulated points per keypoint (one count per point_id that
    // produced >= 1 residual).
    std::vector<bool> pid_seen(N, false);
    for (const auto &r : d.base.residuals) {
        if (r.point_id >= 0 && r.point_id < N && !pid_seen[r.point_id]) {
            pid_seen[r.point_id] = true;
            u32 kp = d.point_labels[r.point_id].kp;
            if ((int)kp < num_nodes) per_kp_n_points[kp]++;
        }
    }
    for (int kp = 0; kp < num_nodes; kp++) {
        if (per_kp_errs[kp].empty()) continue;
        PerKeypointStats s;
        s.kp_id = (u32)kp;
        s.n_obs = (int)per_kp_errs[kp].size();
        s.n_points = per_kp_n_points[kp];
        double sum = 0, sum2 = 0, mx = 0;
        for (double e : per_kp_errs[kp]) {
            sum += e; sum2 += e * e;
            mx = std::max(mx, e);
        }
        s.mean = sum / s.n_obs;
        s.rmse = std::sqrt(sum2 / s.n_obs);
        double var = std::max(0.0, sum2 / s.n_obs - s.mean * s.mean);
        s.std = std::sqrt(var);
        s.max = mx;
        std::vector<double> srt = per_kp_errs[kp];
        std::sort(srt.begin(), srt.end());
        s.median = percentile_sorted(srt, 0.5);
        s.p95 = percentile_sorted(srt, 0.95);
        d.per_keypoint.push_back(s);
    }

    d.success = true;
    return d;
}

// CSV export with frame/keypoint labels decorating each residual.
inline bool save_csv(const Diagnostics &d,
                     const std::vector<std::string> &camera_display_names,
                     const std::string &out_path) {
    std::ofstream f(out_path);
    if (!f.is_open()) return false;
    f << "frame,keypoint,camera,observed_x,observed_y,"
         "reprojected_x,reprojected_y,error_px\n";
    for (const auto &r : d.base.residuals) {
        const PointLabel &lbl = (r.point_id >= 0 &&
                                  r.point_id < (int)d.point_labels.size())
            ? d.point_labels[r.point_id] : PointLabel{};
        std::string cam = (r.camera_idx >= 0 &&
                           r.camera_idx < (int)camera_display_names.size())
            ? camera_display_names[r.camera_idx]
            : std::to_string(r.camera_idx);
        f << lbl.frame << ',' << lbl.kp << ',' << cam << ','
          << r.observed.x() << ',' << r.observed.y() << ','
          << r.reprojected.x() << ',' << r.reprojected.y() << ','
          << r.error_px << '\n';
    }
    return true;
}

} // namespace AnnotationDiagnostics
