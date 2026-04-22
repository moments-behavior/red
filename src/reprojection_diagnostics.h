#pragma once
// reprojection_diagnostics.h — pure, read-only triangulate-and-reproject
// diagnostics for multi-camera calibration.
//
// Given per-camera 2D observations of the same set of N points, and a
// CameraParams per camera, this module triangulates each point (when ≥2
// cameras see it) and reports per-residual error plus per-camera and global
// statistics. No AnnotationMap or project state is mutated — callers can
// invoke this repeatedly without losing the underlying labels.
//
// Supports both telecentric (affine P + radial k1,k2) and perspective
// (full K[R|t] + Brown-Conrady) cameras via CameraParams.telecentric.

#include "camera.h"
#include "red_math.h"
#include <Eigen/Core>
#include <algorithm>
#include <cmath>
#include <fstream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

namespace ReprojectionDiagnostics {

constexpr double UNLABELED_SENTINEL = 1e7;

inline bool is_unlabeled(const Eigen::Vector2d &p) {
    return p.x() > UNLABELED_SENTINEL * 0.9 ||
           p.y() > UNLABELED_SENTINEL * 0.9 ||
           !std::isfinite(p.x()) || !std::isfinite(p.y());
}

struct Residual {
    int point_id = -1;
    int camera_idx = -1;
    Eigen::Vector2d observed = Eigen::Vector2d::Zero();
    Eigen::Vector2d reprojected = Eigen::Vector2d::Zero();
    double error_px = 0.0;
};

struct PerCameraStats {
    std::string name;
    int n_obs = 0;
    double mean = 0, std = 0, median = 0, p95 = 0, max = 0, rmse = 0;
};

enum class Mode {
    Triangulate = 0,   // triangulate 3D from multi-view observations, then reproject
    Known3D     = 1,   // reproject from a supplied set of 3D landmark coordinates
};

inline const char *mode_label(Mode m) {
    return m == Mode::Known3D ? "Known 3D -> reproject"
                              : "Triangulate -> reproject";
}

struct Diagnostics {
    Mode mode = Mode::Triangulate;
    std::vector<Residual> residuals;
    std::vector<PerCameraStats> per_camera;
    std::vector<Eigen::Vector3d> triangulated;   // [n_points_total], NaN = skipped
    int n_points_total = 0;
    int n_points_triangulated = 0;   // or "reprojected" in Known3D mode
    int n_points_skipped = 0;
    double overall_mean = 0, overall_std = 0, overall_median = 0,
           overall_p95 = 0, overall_max = 0, overall_rmse = 0;
    bool success = false;
    std::string error;
};

inline double percentile_sorted(const std::vector<double> &sorted_v, double q) {
    if (sorted_v.empty()) return 0.0;
    double idx = q * (sorted_v.size() - 1);
    size_t lo = (size_t)std::floor(idx);
    size_t hi = (size_t)std::ceil(idx);
    double t = idx - lo;
    return sorted_v[lo] * (1.0 - t) + sorted_v[hi] * t;
}

// Core primitive: triangulate (or take known 3D) + reproject + residuals. Pure.
//   observations[m][i] = 2D observation of point i in camera m (sentinel = unlabeled)
//   cameras[m] = camera parameters (branches on .telecentric)
//   camera_names[m] = display name
//   known_3d = optional. If non-empty, used as the 3D point for index i when
//              known_3d[i] is finite; triangulation is skipped. If empty, all
//              points are triangulated from their multi-view observations.
inline Diagnostics compute_diagnostics(
    const std::vector<std::vector<Eigen::Vector2d>> &observations,
    const std::vector<CameraParams> &cameras,
    const std::vector<std::string> &camera_names,
    const std::vector<Eigen::Vector3d> &known_3d = {}) {

    Diagnostics d;
    d.mode = known_3d.empty() ? Mode::Triangulate : Mode::Known3D;
    int M = (int)cameras.size();
    if (M == 0 || (int)observations.size() != M) {
        d.error = "observations / cameras length mismatch";
        return d;
    }
    bool telecentric = cameras[0].telecentric;

    int N = 0;
    for (const auto &v : observations) N = std::max(N, (int)v.size());
    if (!known_3d.empty()) N = std::max(N, (int)known_3d.size());
    d.n_points_total = N;
    d.triangulated.assign(N,
        Eigen::Vector3d(std::nan(""), std::nan(""), std::nan("")));

    d.per_camera.resize(M);
    for (int m = 0; m < M; m++) {
        d.per_camera[m].name = m < (int)camera_names.size()
            ? camera_names[m] : ("Cam" + std::to_string(m));
    }

    std::vector<std::vector<double>> per_cam_errs(M);
    std::vector<double> all_errs;
    all_errs.reserve((size_t)N * M);

    for (int i = 0; i < N; i++) {
        std::vector<Eigen::Vector2d> undist_pts;
        std::vector<Eigen::Matrix<double, 3, 4>> proj_mats;
        std::vector<int> contributing;
        contributing.reserve(M);

        for (int m = 0; m < M; m++) {
            if (i >= (int)observations[m].size()) continue;
            const auto &obs = observations[m][i];
            if (is_unlabeled(obs)) continue;

            Eigen::Vector2d undist;
            if (telecentric) {
                undist = red_math::undistortPointTelecentric(
                    obs, cameras[m].k, cameras[m].dist_coeffs);
            } else {
                undist = red_math::undistortPoint(
                    obs, cameras[m].k, cameras[m].dist_coeffs);
            }
            undist_pts.push_back(undist);
            proj_mats.push_back(cameras[m].projection_mat);
            contributing.push_back(m);
        }

        Eigen::Vector3d pt3d(std::nan(""), std::nan(""), std::nan(""));
        if (d.mode == Mode::Known3D) {
            // Need the known 3D and at least one observation.
            if (i >= (int)known_3d.size() ||
                !std::isfinite(known_3d[i].x()) ||
                !std::isfinite(known_3d[i].y()) ||
                !std::isfinite(known_3d[i].z()) ||
                contributing.empty()) {
                d.n_points_skipped++;
                continue;
            }
            pt3d = known_3d[i];
        } else {
            if (contributing.size() < 2) {
                d.n_points_skipped++;
                continue;
            }
            pt3d = red_math::triangulatePoints(undist_pts, proj_mats);
            if (!std::isfinite(pt3d.x()) || !std::isfinite(pt3d.y()) ||
                !std::isfinite(pt3d.z())) {
                d.n_points_skipped++;
                continue;
            }
        }
        d.triangulated[i] = pt3d;
        d.n_points_triangulated++;

        for (int m : contributing) {
            Eigen::Vector2d reproj;
            if (telecentric) {
                reproj = red_math::projectPointTelecentric(
                    pt3d, cameras[m].projection_mat,
                    cameras[m].k, cameras[m].dist_coeffs);
            } else {
                reproj = red_math::projectPointR(
                    pt3d, cameras[m].r, cameras[m].tvec,
                    cameras[m].k, cameras[m].dist_coeffs);
            }
            double err = (reproj - observations[m][i]).norm();

            Residual r;
            r.point_id = i;
            r.camera_idx = m;
            r.observed = observations[m][i];
            r.reprojected = reproj;
            r.error_px = err;
            d.residuals.push_back(r);

            per_cam_errs[m].push_back(err);
            all_errs.push_back(err);
        }
    }

    for (int m = 0; m < M; m++) {
        auto &pc = d.per_camera[m];
        pc.n_obs = (int)per_cam_errs[m].size();
        if (pc.n_obs == 0) continue;
        double sum = 0, sum2 = 0, max_e = 0;
        for (double e : per_cam_errs[m]) {
            sum += e; sum2 += e * e;
            max_e = std::max(max_e, e);
        }
        pc.mean = sum / pc.n_obs;
        pc.rmse = std::sqrt(sum2 / pc.n_obs);
        // Population s.d. (divide by N) to stay consistent with RMSE.
        double var = std::max(0.0, sum2 / pc.n_obs - pc.mean * pc.mean);
        pc.std = std::sqrt(var);
        pc.max = max_e;
        std::vector<double> s = per_cam_errs[m];
        std::sort(s.begin(), s.end());
        pc.median = percentile_sorted(s, 0.5);
        pc.p95 = percentile_sorted(s, 0.95);
    }

    if (!all_errs.empty()) {
        double sum = 0, sum2 = 0, max_e = 0;
        for (double e : all_errs) {
            sum += e; sum2 += e * e;
            max_e = std::max(max_e, e);
        }
        d.overall_mean = sum / all_errs.size();
        d.overall_rmse = std::sqrt(sum2 / all_errs.size());
        double var = std::max(0.0,
            sum2 / all_errs.size() - d.overall_mean * d.overall_mean);
        d.overall_std = std::sqrt(var);
        d.overall_max = max_e;
        std::sort(all_errs.begin(), all_errs.end());
        d.overall_median = percentile_sorted(all_errs, 0.5);
        d.overall_p95 = percentile_sorted(all_errs, 0.95);
    }

    d.success = true;
    return d;
}

// ── CSV helpers ──────────────────────────────────────────────────────────

// Parse the Cam<serial>.csv 2D label format used by TelecentricDLT.
//   Header: "Target"
//   Rows:   "point_id,kp_id,x,y"   (sentinel = UNLABELED_SENTINEL)
// flip_y mirrors TelecentricDLT::parse_2d_labels so coordinates stay
// consistent with the system the DLT was fit in.
inline std::vector<Eigen::Vector2d> parse_observations_csv(
    const std::string &path, int image_height, bool flip_y) {
    std::vector<Eigen::Vector2d> pts;
    std::ifstream f(path);
    if (!f.is_open()) return pts;
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        int pid;
        double x, y;
        if (!std::getline(ss, tok, ',')) continue; pid = std::stoi(tok);
        if (!std::getline(ss, tok, ',')) continue; (void)std::stoi(tok); // kp_id
        if (!std::getline(ss, tok, ',')) continue; x = std::stod(tok);
        if (!std::getline(ss, tok, ',')) continue; y = std::stod(tok);
        if (pid >= (int)pts.size())
            pts.resize(pid + 1,
                Eigen::Vector2d(UNLABELED_SENTINEL, UNLABELED_SENTINEL));
        if (x > UNLABELED_SENTINEL * 0.9 || y > UNLABELED_SENTINEL * 0.9) {
            pts[pid] = Eigen::Vector2d(UNLABELED_SENTINEL, UNLABELED_SENTINEL);
        } else {
            if (flip_y) y = image_height - y;
            pts[pid] = Eigen::Vector2d(x, y);
        }
    }
    return pts;
}

// Parse a landmarks 3D CSV of the form "x,y,z\nX,Y,Z\n...".
inline std::vector<Eigen::Vector3d>
parse_landmarks_3d_csv(const std::string &path) {
    std::vector<Eigen::Vector3d> pts;
    std::ifstream f(path);
    if (!f.is_open()) return pts;
    std::string line;
    std::getline(f, line); // header
    while (std::getline(f, line)) {
        if (line.empty()) continue;
        std::istringstream ss(line);
        std::string tok;
        double x, y, z;
        if (!std::getline(ss, tok, ',')) continue; x = std::stod(tok);
        if (!std::getline(ss, tok, ',')) continue; y = std::stod(tok);
        if (!std::getline(ss, tok, ',')) continue; z = std::stod(tok);
        pts.emplace_back(x, y, z);
    }
    return pts;
}

// Load saved DLT coefficients + 2D labels and compute diagnostics.
// Matches the folder layout of TelecentricDLT::save_dlt_coefficients.
//   dlt_folder/Cam<serial>_dlt.csv (+ optional Cam<serial>_distortion.csv)
//   labels_folder/Cam<serial>.csv
// If known_3d_csv is non-empty, it is parsed and the diagnostic runs in
// Known3D mode (no triangulation).
inline Diagnostics compute_from_disk(
    const std::vector<std::string> &camera_serials,
    const std::string &dlt_folder,
    const std::string &labels_folder,
    int image_height,
    bool flip_y,
    const std::string &known_3d_csv = "") {

    Diagnostics d;
    int M = (int)camera_serials.size();
    if (M == 0) { d.error = "no cameras"; return d; }

    std::vector<CameraParams> cams(M);
    std::vector<std::string> names(M);
    std::vector<std::vector<Eigen::Vector2d>> obs(M);

    for (int m = 0; m < M; m++) {
        names[m] = "Cam" + camera_serials[m];
        std::string dlt_path =
            dlt_folder + "/Cam" + camera_serials[m] + "_dlt.csv";
        std::string err;
        if (!camera_load_params_from_dlt_csv(dlt_path, cams[m], err)) {
            d.error = err; return d;
        }
        std::string lbl_path =
            labels_folder + "/Cam" + camera_serials[m] + ".csv";
        obs[m] = parse_observations_csv(lbl_path, image_height, flip_y);
        if (obs[m].empty()) {
            d.error = "no labels in " + lbl_path; return d;
        }
    }

    std::vector<Eigen::Vector3d> known_3d;
    if (!known_3d_csv.empty()) {
        known_3d = parse_landmarks_3d_csv(known_3d_csv);
        if (known_3d.empty()) {
            d.error = "no 3D points in " + known_3d_csv;
            return d;
        }
    }

    return compute_diagnostics(obs, cams, names, known_3d);
}

// Write residuals to CSV for downstream plotting/analysis.
inline bool save_csv(const Diagnostics &d, const std::string &out_path) {
    std::ofstream f(out_path);
    if (!f.is_open()) return false;
    f << "point_id,camera,observed_x,observed_y,"
         "reprojected_x,reprojected_y,error_px\n";
    for (const auto &r : d.residuals) {
        const std::string &cn =
            (r.camera_idx >= 0 && r.camera_idx < (int)d.per_camera.size())
                ? d.per_camera[r.camera_idx].name
                : std::to_string(r.camera_idx);
        f << r.point_id << ',' << cn << ','
          << r.observed.x() << ',' << r.observed.y() << ','
          << r.reprojected.x() << ',' << r.reprojected.y() << ','
          << r.error_px << '\n';
    }
    return true;
}

} // namespace ReprojectionDiagnostics
