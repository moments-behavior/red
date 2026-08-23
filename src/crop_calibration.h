#pragma once
// Crop-transform tool for two-stage cropped-sensor calibration.
//
// Stage 1 calibrates the rig at full sensor resolution (ChArUco). When the
// cameras are then switched to a sensor ROI crop (same optics), the
// calibration transforms exactly: the crop is a pure pixel translation, so
// only the principal point changes (cx -= offset_x, cy -= offset_y).
// Focal lengths, distortion (defined about the principal point in normalized
// coordinates), and extrinsics are copied verbatim.
//
// Also holds the posts (fixed-landmark) I/O shared between the GUI and tests:
// posts are hand-clicked at full frame, triangulated with the stage-1
// calibration, and stored in posts_3d.csv for the stage-2 refinement.
//
// NOTE: when posts are viewed through plexiglass, the triangulated positions
// are *apparent* (refracted) positions in the stage-1 frame. This is
// intentional and self-consistent: all subsequent recordings look through the
// same glass, so do NOT "correct" these coordinates for refraction.

#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <map>
#include <string>
#include <vector>

#include <Eigen/Dense>

#include "json.hpp"
#include "opencv_yaml_io.h"
#include "calibration_pipeline.h"
#include "red_math.h"

namespace CropCalibration {

// ─────────────────────────────────────────────────────────────────────────────
// Crop spec (crop_info.json)
// ─────────────────────────────────────────────────────────────────────────────

struct CameraCrop {
    std::string serial;
    int offset_x = 0;   // OffsetX of the ROI on the full sensor
    int offset_y = 0;   // OffsetY of the ROI on the full sensor
    int width = 0;      // cropped image width
    int height = 0;     // cropped image height
};

struct CropSpec {
    std::string source_calibration;  // provenance: stage-1 calibration folder
    std::string timestamp;           // when the crop was applied
    std::vector<CameraCrop> cameras;
};

inline void to_json(nlohmann::json &j, const CameraCrop &c) {
    j = nlohmann::json{{"serial", c.serial},
                       {"offset_x", c.offset_x},
                       {"offset_y", c.offset_y},
                       {"width", c.width},
                       {"height", c.height}};
}

inline void from_json(const nlohmann::json &j, CameraCrop &c) {
    c.serial = j.value("serial", std::string());
    c.offset_x = j.value("offset_x", 0);
    c.offset_y = j.value("offset_y", 0);
    c.width = j.value("width", 0);
    c.height = j.value("height", 0);
}

inline void to_json(nlohmann::json &j, const CropSpec &s) {
    j = nlohmann::json{{"source_calibration", s.source_calibration},
                       {"timestamp", s.timestamp},
                       {"cameras", s.cameras}};
}

inline void from_json(const nlohmann::json &j, CropSpec &s) {
    s.source_calibration = j.value("source_calibration", std::string());
    s.timestamp = j.value("timestamp", std::string());
    s.cameras.clear();
    if (j.contains("cameras"))
        s.cameras = j.at("cameras").get<std::vector<CameraCrop>>();
}

inline bool load_crop_spec(const std::string &path, CropSpec &spec,
                           std::string *err = nullptr) {
    std::ifstream f(path);
    if (!f.is_open()) {
        if (err) *err = "Cannot open " + path;
        return false;
    }
    try {
        nlohmann::json j;
        f >> j;
        spec = j.get<CropSpec>();
    } catch (const std::exception &e) {
        if (err) *err = "Parse error in " + path + ": " + e.what();
        return false;
    }
    return true;
}

inline bool save_crop_spec(const CropSpec &spec, const std::string &path,
                           std::string *err = nullptr) {
    std::ofstream f(path);
    if (!f.is_open()) {
        if (err) *err = "Cannot write " + path;
        return false;
    }
    nlohmann::json j = spec;
    f << j.dump(2) << "\n";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Calibration folder resolution
// ─────────────────────────────────────────────────────────────────────────────

// Users may hand us either a timestamped calibration folder (contains
// Cam<serial>.yaml directly) or the aruco output root that holds timestamped
// YYYY_MM_DD_* subfolders. Resolve to the folder that actually has the files.
inline std::string resolve_calibration_folder(const std::string &folder) {
    namespace fs = std::filesystem;
    if (!fs::is_directory(folder)) return folder;

    auto has_calib_files = [](const std::string &dir) {
        namespace fs = std::filesystem;
        if (fs::exists(dir + "/calibration_data.json")) return true;
        std::error_code ec;
        for (const auto &entry : fs::directory_iterator(dir, ec)) {
            std::string name = entry.path().filename().string();
            if (name.rfind("Cam", 0) == 0 &&
                entry.path().extension() == ".yaml")
                return true;
        }
        return false;
    };

    if (has_calib_files(folder)) return folder;

    std::string latest;
    std::error_code ec;
    for (const auto &entry : fs::directory_iterator(folder, ec)) {
        if (!entry.is_directory()) continue;
        std::string name = entry.path().filename().string();
        // Timestamped folders look like YYYY_MM_DD[_HH_MM_SS]
        if (name.size() >= 10 && name[4] == '_' && name > latest)
            latest = name;
    }
    if (!latest.empty()) return folder + "/" + latest;
    return folder;
}

// ─────────────────────────────────────────────────────────────────────────────
// Crop transform
// ─────────────────────────────────────────────────────────────────────────────

struct CropResult {
    bool success = false;
    std::string error;
    std::string output_folder;
    std::vector<std::string> warnings;
};

// Derive a cropped-frame calibration from a stage-1 (full-frame) calibration.
// For each camera in the spec: cx -= offset_x, cy -= offset_y, image dims set
// to the crop size; K (focal), distortion, R, t copied verbatim.
inline CropResult apply_crop_to_calibration(const std::string &stage1_folder,
                                            const CropSpec &spec,
                                            const std::string &output_folder) {
    namespace fs = std::filesystem;
    CropResult result;

    std::string src = resolve_calibration_folder(stage1_folder);
    if (spec.cameras.empty()) {
        result.error = "Crop spec has no cameras";
        return result;
    }

    std::vector<CalibrationPipeline::CameraPose> poses;
    std::vector<std::string> cam_names;
    std::vector<int> widths, heights;

    for (const auto &crop : spec.cameras) {
        std::string yaml_path = src + "/Cam" + crop.serial + ".yaml";
        if (!fs::exists(yaml_path)) {
            result.error = "Missing calibration file: " + yaml_path;
            return result;
        }
        if (crop.width <= 0 || crop.height <= 0) {
            result.error = "Camera " + crop.serial +
                           ": crop width/height must be positive";
            return result;
        }
        try {
            auto yf = opencv_yaml::read(yaml_path);
            CalibrationPipeline::CameraPose pose;
            pose.K = yf.getMatrix("camera_matrix").block<3, 3>(0, 0);
            Eigen::MatrixXd dist_mat = yf.getMatrix("distortion_coefficients");
            for (int j = 0; j < 5; j++) pose.dist(j) = dist_mat(j, 0);
            pose.R = yf.getMatrix("rc_ext").block<3, 3>(0, 0);
            Eigen::MatrixXd t_mat = yf.getMatrix("tc_ext");
            pose.t = Eigen::Vector3d(t_mat(0, 0), t_mat(1, 0), t_mat(2, 0));

            int full_w = yf.getInt("image_width");
            int full_h = yf.getInt("image_height");
            if (full_w > 0 && full_h > 0 &&
                (crop.offset_x + crop.width > full_w ||
                 crop.offset_y + crop.height > full_h)) {
                result.warnings.push_back(
                    "Camera " + crop.serial + ": crop (" +
                    std::to_string(crop.offset_x) + "," +
                    std::to_string(crop.offset_y) + " " +
                    std::to_string(crop.width) + "x" +
                    std::to_string(crop.height) +
                    ") extends beyond full frame " + std::to_string(full_w) +
                    "x" + std::to_string(full_h) +
                    " recorded in the YAML — check offsets");
            }

            // The exact crop transform: pure pixel translation.
            pose.K(0, 2) -= crop.offset_x;
            pose.K(1, 2) -= crop.offset_y;

            poses.push_back(pose);
            cam_names.push_back(crop.serial);
            widths.push_back(crop.width);
            heights.push_back(crop.height);
        } catch (const std::exception &e) {
            result.error = "Error reading " + yaml_path + ": " + e.what();
            return result;
        }
    }

    std::string write_status;
    if (!CalibrationPipeline::write_calibration(poses, cam_names, output_folder,
                                                widths, heights,
                                                &write_status)) {
        result.error = write_status;
        return result;
    }

    // Provenance
    CropSpec out_spec = spec;
    out_spec.source_calibration = src;
    if (out_spec.timestamp.empty()) {
        std::time_t t = std::time(nullptr);
        std::tm tstruct;
#ifdef _WIN32
        localtime_s(&tstruct, &t);
#else
        localtime_r(&t, &tstruct);
#endif
        char tbuf[64];
        std::strftime(tbuf, sizeof(tbuf), "%Y_%m_%d_%H_%M_%S", &tstruct);
        out_spec.timestamp = tbuf;
    }
    std::string spec_err;
    if (!save_crop_spec(out_spec, output_folder + "/crop_info.json", &spec_err))
        result.warnings.push_back(spec_err);

    // Copy the calibration database forward so load_calibration_from_folder
    // still works on the cropped folder. Note: board poses / residuals inside
    // it remain in FULL-FRAME pixel coordinates.
    std::error_code ec;
    if (fs::exists(src + "/calibration_data.json"))
        fs::copy_file(src + "/calibration_data.json",
                      output_folder + "/calibration_data.json",
                      fs::copy_options::overwrite_existing, ec);

    result.success = true;
    result.output_folder = output_folder;
    return result;
}

// Shift full-frame observations (OpenCV y-down pixel coords) into crop-local
// coordinates: p' = p - (offset_x, offset_y). Observations landing outside
// their camera's crop are dropped and reported in `dropped` — those posts
// would not be visible in the real cropped recording. Cameras absent from the
// spec are dropped entirely (with a report).
inline std::map<std::string, std::map<int, Eigen::Vector2d>>
shift_landmarks_to_crop(
    const std::map<std::string, std::map<int, Eigen::Vector2d>> &full,
    const CropSpec &spec, std::vector<std::string> &dropped) {
    std::map<std::string, const CameraCrop *> by_serial;
    for (const auto &c : spec.cameras) by_serial[c.serial] = &c;

    std::map<std::string, std::map<int, Eigen::Vector2d>> out;
    for (const auto &[serial, pts] : full) {
        auto it = by_serial.find(serial);
        if (it == by_serial.end()) {
            dropped.push_back("Camera " + serial +
                              ": no crop in spec — all observations dropped");
            continue;
        }
        const CameraCrop &c = *it->second;
        for (const auto &[pid, px] : pts) {
            Eigen::Vector2d p(px.x() - c.offset_x, px.y() - c.offset_y);
            if (p.x() < 0 || p.y() < 0 || p.x() >= c.width ||
                p.y() >= c.height) {
                dropped.push_back("Post " + std::to_string(pid) + " in Cam" +
                                  serial + " falls outside the crop");
                continue;
            }
            out[serial][pid] = p;
        }
    }
    return out;
}

// ─────────────────────────────────────────────────────────────────────────────
// Posts: triangulation with per-post quality report, and posts_3d.csv I/O
// ─────────────────────────────────────────────────────────────────────────────

struct PostReport {
    int id = 0;                 // post id == skeleton node index
    int n_views = 0;            // cameras with a click for this post
    double mean_reproj = 0.0;   // px, across contributing cameras (refined)
    double max_reproj = 0.0;    // px
    double dlt_mean_reproj = 0.0;  // px, before nonlinear refinement
    std::map<std::string, double> per_cam_reproj;
    // Cameras whose click lies beyond the stage-1 calibration's observed
    // radius — the distortion model is EXTRAPOLATED there, so a large
    // residual means "uncalibrated image region", not "bad click".
    std::vector<std::string> extrapolated_cams;
    bool accepted = false;      // triangulated and mean_reproj <= max_reproj threshold
};

// Reprojection cost for refining a single post's 3D position (cameras fixed).
struct PostPointCost {
    double obs_x, obs_y;
    Eigen::Matrix3d R;
    Eigen::Vector3d t;
    Eigen::Matrix3d K;
    Eigen::Matrix<double, 5, 1> dist;
    PostPointCost(const Eigen::Vector2d &obs,
                  const CalibrationPipeline::CameraPose &cam)
        : obs_x(obs.x()), obs_y(obs.y()), R(cam.R), t(cam.t), K(cam.K),
          dist(cam.dist) {}
    template <typename T>
    bool operator()(const T *X, T *res) const {
        T p[3];
        for (int i = 0; i < 3; i++)
            p[i] = T(R(i, 0)) * X[0] + T(R(i, 1)) * X[1] + T(R(i, 2)) * X[2] +
                   T(t(i));
        T xp = p[0] / p[2], yp = p[1] / p[2];
        T r2 = xp * xp + yp * yp, r4 = r2 * r2, r6 = r4 * r2;
        T radial = T(1) + T(dist(0)) * r2 + T(dist(1)) * r4 + T(dist(4)) * r6;
        T xpp = xp * radial + T(2) * T(dist(2)) * xp * yp +
                T(dist(3)) * (r2 + T(2) * xp * xp);
        T ypp = yp * radial + T(dist(2)) * (r2 + T(2) * yp * yp) +
                T(2) * T(dist(3)) * xp * yp;
        res[0] = T(K(0, 0)) * xpp + T(K(0, 2)) - T(obs_x);
        res[1] = T(K(1, 1)) * ypp + T(K(1, 2)) - T(obs_y);
        return true;
    }
};

// Triangulate hand-clicked posts (landmarks: cam serial -> post id -> pixel,
// OpenCV y-down) with the stage-1 calibration: linear DLT for the initial
// estimate, then per-post robust nonlinear refinement (Huber) — the DLT
// minimizes an algebraic proxy and weights all views equally, so one bad
// view (defocus, misclick) drags the whole post; the robust ML refinement
// downweights it instead. Per-post per-camera residuals are reported so bad
// clicks are caught. Posts with < 2 views are reported but not triangulated.
// All triangulated posts land in posts_3d; `accepted` flags the threshold.
// coverage_radius_px (optional, parallel to cam_names, <=0 = unknown): the
// stage-1 calibration's maximum observed radius per camera; clicks beyond it
// are flagged as extrapolated.
inline std::vector<PostReport> triangulate_posts_with_report(
    const std::vector<std::string> &cam_names,
    const std::map<std::string, std::map<int, Eigen::Vector2d>> &landmarks,
    const std::vector<CalibrationPipeline::CameraPose> &poses,
    double max_reproj,
    std::map<int, Eigen::Vector3d> &posts_3d,
    const std::vector<double> &coverage_radius_px = {}) {

    posts_3d.clear();
    int nc = (int)cam_names.size();

    // Group observations by post id
    std::map<int, std::vector<std::pair<int, Eigen::Vector2d>>> pobs;
    for (int c = 0; c < nc; c++) {
        auto it = landmarks.find(cam_names[c]);
        if (it == landmarks.end()) continue;
        for (const auto &[pid, px] : it->second)
            pobs[pid].push_back({c, px});
    }

    std::vector<PostReport> reports;
    for (const auto &[pid, obs] : pobs) {
        PostReport rep;
        rep.id = pid;
        rep.n_views = (int)obs.size();
        if (rep.n_views < 2) {
            reports.push_back(rep);
            continue;
        }

        std::vector<Eigen::Vector2d> pts_undist;
        std::vector<Eigen::Matrix<double, 3, 4>> Ps;
        for (const auto &[ci, px] : obs) {
            pts_undist.push_back(
                red_math::undistortPoint(px, poses[ci].K, poses[ci].dist));
            Ps.push_back(red_math::projectionFromKRt(poses[ci].K, poses[ci].R,
                                                     poses[ci].t));
        }
        Eigen::Vector3d X = red_math::triangulatePoints(pts_undist, Ps);
        if (!X.allFinite()) {
            reports.push_back(rep);
            continue;
        }

        // MSVC rejects capturing a structured binding in a lambda under
        // C++17 (error C3493), so alias `obs` before capturing it.
        const auto &obs_ref = obs;
        auto mean_err = [&](const Eigen::Vector3d &pt) {
            double s = 0.0;
            for (const auto &[ci, px] : obs_ref) {
                auto rv = red_math::rotationMatrixToVector(poses[ci].R);
                s += (red_math::projectPoint(pt, rv, poses[ci].t, poses[ci].K,
                                             poses[ci].dist) - px).norm();
            }
            return s / obs_ref.size();
        };
        rep.dlt_mean_reproj = mean_err(X);

        // Robust nonlinear refinement of X (cameras fixed)
        {
            double Xp[3] = {X.x(), X.y(), X.z()};
            ceres::Problem problem;
            for (const auto &[ci, px] : obs)
                problem.AddResidualBlock(
                    new ceres::AutoDiffCostFunction<PostPointCost, 2, 3>(
                        new PostPointCost(px, poses[ci])),
                    new ceres::HuberLoss(2.0), Xp);
            ceres::Solver::Options opt;
            opt.linear_solver_type = ceres::DENSE_QR;
            opt.max_num_iterations = 50;
            opt.minimizer_progress_to_stdout = false;
            ceres::Solver::Summary sum;
            ceres::Solve(opt, &problem, &sum);
            Eigen::Vector3d Xr(Xp[0], Xp[1], Xp[2]);
            if (Xr.allFinite() && sum.IsSolutionUsable() &&
                mean_err(Xr) <= rep.dlt_mean_reproj)
                X = Xr;
        }

        double sum = 0.0;
        for (const auto &[ci, px] : obs) {
            auto rv = red_math::rotationMatrixToVector(poses[ci].R);
            auto pr = red_math::projectPoint(X, rv, poses[ci].t, poses[ci].K,
                                             poses[ci].dist);
            double e = (pr - px).norm();
            rep.per_cam_reproj[cam_names[ci]] = e;
            rep.max_reproj = std::max(rep.max_reproj, e);
            sum += e;
            if (ci < (int)coverage_radius_px.size() &&
                coverage_radius_px[ci] > 0) {
                double r = (px - Eigen::Vector2d(poses[ci].K(0, 2),
                                                 poses[ci].K(1, 2))).norm();
                if (r > coverage_radius_px[ci])
                    rep.extrapolated_cams.push_back(cam_names[ci]);
            }
        }
        rep.mean_reproj = sum / obs.size();
        rep.accepted = rep.mean_reproj <= max_reproj;
        posts_3d[pid] = X;
        reports.push_back(rep);
    }
    return reports;
}

// Per-camera maximum observed corner radius (px from the principal point) of
// a stage-1 calibration — from calibration_data.json's residual pixel
// coordinates. The distortion model is only CONSTRAINED inside this radius;
// beyond it, it extrapolates. Returns empty when the data is unavailable
// (old runs without obs_x/obs_y export).
inline std::vector<double> load_coverage_radii(
    const std::string &stage1_folder,
    const std::vector<std::string> &cam_names,
    const std::vector<CalibrationPipeline::CameraPose> &poses) {
    std::string src = resolve_calibration_folder(stage1_folder);
    std::ifstream f(src + "/calibration_data.json");
    if (!f.is_open()) return {};
    nlohmann::json j;
    std::vector<std::string> run_cams;
    std::vector<int> cam_idx;
    std::vector<double> obs_x, obs_y;
    // Whole parse inside the try: a malformed/partial file (wrong types,
    // obs_x without obs_y) must degrade to "no coverage data", not throw
    // into the calling button handler.
    try {
        f >> j;
        if (!j.contains("residuals") || !j.contains("camera_names"))
            return {};
        const auto &r = j["residuals"];
        if (!r.contains("obs_x") || !r.contains("obs_y") ||
            !r.contains("camera_idx"))
            return {};
        run_cams = j["camera_names"].get<std::vector<std::string>>();
        cam_idx = r["camera_idx"].get<std::vector<int>>();
        obs_x = r["obs_x"].get<std::vector<double>>();
        obs_y = r["obs_y"].get<std::vector<double>>();
    } catch (...) {
        return {};
    }
    if (obs_x.size() != cam_idx.size() || obs_y.size() != cam_idx.size())
        return {};

    std::vector<double> radii(cam_names.size(), 0.0);
    for (size_t c = 0; c < cam_names.size(); c++) {
        int ri = -1;
        for (size_t k = 0; k < run_cams.size(); k++)
            if (run_cams[k] == cam_names[c]) { ri = (int)k; break; }
        if (ri < 0 || c >= poses.size()) continue;
        double cx = poses[c].K(0, 2), cy = poses[c].K(1, 2), rmax = 0.0;
        for (size_t k = 0; k < cam_idx.size(); k++) {
            if (cam_idx[k] != ri) continue;
            rmax = std::max(rmax, std::hypot(obs_x[k] - cx, obs_y[k] - cy));
        }
        radii[c] = rmax;
    }
    return radii;
}

// posts_3d.csv: header "x,y,z", row index == post id (skeleton node index).
// Posts that could not be triangulated are written as nan,nan,nan so row
// indices stay aligned. Compatible with TelecentricDLT::parse_3d_landmarks
// and ReprojectionDiagnostics::parse_landmarks_3d_csv (Known-3D mode).
inline bool write_posts_3d_csv(const std::string &path, int n_posts,
                               const std::map<int, Eigen::Vector3d> &posts_3d,
                               std::string *err = nullptr) {
    std::ofstream f(path);
    if (!f.is_open()) {
        if (err) *err = "Cannot write " + path;
        return false;
    }
    f << "x,y,z\n";
    char buf[128];
    for (int i = 0; i < n_posts; i++) {
        auto it = posts_3d.find(i);
        if (it == posts_3d.end()) {
            f << "nan,nan,nan\n";
        } else {
            snprintf(buf, sizeof(buf), "%.6f,%.6f,%.6f\n", it->second.x(),
                     it->second.y(), it->second.z());
            f << buf;
        }
    }
    return true;
}

// Returns post id -> 3D position; rows with non-finite values are skipped.
inline std::map<int, Eigen::Vector3d> read_posts_3d_csv(
    const std::string &path) {
    std::map<int, Eigen::Vector3d> posts;
    std::ifstream f(path);
    if (!f.is_open()) return posts;
    std::string line;
    if (!std::getline(f, line)) return posts;  // header
    int idx = 0;
    while (std::getline(f, line)) {
        double x, y, z;
        if (sscanf(line.c_str(), "%lf,%lf,%lf", &x, &y, &z) == 3 &&
            std::isfinite(x) && std::isfinite(y) && std::isfinite(z))
            posts[idx] = Eigen::Vector3d(x, y, z);
        idx++;
    }
    return posts;
}

inline bool write_posts_report_json(const std::string &path,
                                    const std::vector<PostReport> &reports,
                                    std::string *err = nullptr) {
    nlohmann::json arr = nlohmann::json::array();
    for (const auto &r : reports) {
        nlohmann::json j{{"id", r.id},
                         {"n_views", r.n_views},
                         {"mean_reproj", r.mean_reproj},
                         {"max_reproj", r.max_reproj},
                         {"dlt_mean_reproj", r.dlt_mean_reproj},
                         {"accepted", r.accepted}};
        j["per_cam_reproj"] = r.per_cam_reproj;
        if (!r.extrapolated_cams.empty())
            j["extrapolated_cams"] = r.extrapolated_cams;
        arr.push_back(j);
    }
    std::ofstream f(path);
    if (!f.is_open()) {
        if (err) *err = "Cannot write " + path;
        return false;
    }
    f << arr.dump(2) << "\n";
    return true;
}

}  // namespace CropCalibration
