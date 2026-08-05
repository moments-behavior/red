#pragma once
// Stage-2 refinement for the two-stage cropped-sensor calibration.
//
// Inputs: a cropped-frame calibration (stage-1 calibration passed through
// CropCalibration::apply_crop_to_calibration) and fixed 3D posts
// (posts_3d.csv, triangulated at full frame with the stage-1 calibration).
// Posts are hand-clicked in the cropped views; this solver refines ONLY
// per-camera intrinsics against them:
//
//   - Extrinsics (R, t) are locked BY CONSTRUCTION: they are constants baked
//     into the cost functor, never Ceres parameter blocks. Same for the post
//     3D positions and the distortion coefficients.
//   - Free parameters per camera: principal point cx, cy (absorbs pointing
//     drift — in a narrow crop a small rotation is nearly degenerate with a
//     principal-point shift) and optionally focal length (fx=fy tie via the
//     stage-1 fy/fx ratio; absorbs the plexiglass apparent-depth shift).
//   - Priors pull toward the crop-shifted stage-1 values.
//
// With extrinsics and points fixed the cameras decouple into independent
// 3-parameter problems, so this deliberately does NOT reuse the 15-param
// bundle-adjustment machinery. If drift exceeds ~tens of px, re-run stage 1
// instead of loosening the priors.

#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <random>
#include <set>
#include <string>
#include <vector>

#include <Eigen/Dense>
#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include "json.hpp"
#include "opencv_yaml_io.h"
#include "calibration_pipeline.h"
#include "crop_calibration.h"
#include "red_math.h"

namespace CroppedRefinement {
namespace fs = std::filesystem;

struct RefineConfig {
    std::string calibration_folder;  // CROPPED (shifted) calibration from the crop transform
    std::string posts_3d_csv;        // fixed 3D posts from stage 1b
    std::string output_folder;
    std::vector<std::string> camera_names;

    bool free_focal = false;              // false: only cx,cy free; true: also f (fx=fy tie)
    double prior_principal_weight = 0.05; // residual = w*(c - c0), px per px of deviation
    double prior_focal_weight = 0.5;      // stronger: f barely observable from a shallow volume
    double huber_delta = 2.0;             // robust loss on reprojection residuals
    double outlier_th = 25.0;             // drop obs whose INITIAL reproj exceeds this (bad clicks)
    int max_iter = 100;

    double holdout_fraction = 0.0;  // split by post id; 0 = disabled
    uint32_t holdout_seed = 42;
};

struct CamRefineStats {
    std::string serial;
    int n_obs = 0;  // train observations used in the solve
    double reproj_before = 0.0;
    double reproj_after = 0.0;
    double d_cx = 0.0, d_cy = 0.0, d_f = 0.0;
};

struct RefineResult {
    bool success = false;
    std::string error;
    std::vector<std::string> warnings;

    double mean_before = 0.0;    // train obs, px
    double mean_after = 0.0;
    double holdout_before = 0.0; // held-out posts, direct projection
    double holdout_after = 0.0;
    int holdout_obs = 0;
    int obs_used = 0;
    int obs_dropped = 0;

    std::vector<CamRefineStats> per_camera;
    std::string output_folder;
};

// Reprojection of a FIXED 3D post through a camera whose extrinsics and
// distortion are FIXED; only intr = {f, cx, cy} is a parameter block.
struct PostReprojectionCost {
    double X[3];       // post position (world)
    double rvec[3];    // fixed extrinsics
    double tvec[3];
    double dist[5];    // fixed Brown-Conrady k1,k2,p1,p2,k3
    double fy_ratio;   // stage-1 fy/fx, preserves aspect under the fx=fy tie
    double obs_x, obs_y;

    template <typename T>
    bool operator()(const T *intr, T *residuals) const {
        // Rotate + translate (constants promoted to T)
        T rv[3] = {T(rvec[0]), T(rvec[1]), T(rvec[2])};
        T pt[3] = {T(X[0]), T(X[1]), T(X[2])};
        T p[3];
        ceres::AngleAxisRotatePoint(rv, pt, p);
        p[0] += T(tvec[0]);
        p[1] += T(tvec[1]);
        p[2] += T(tvec[2]);

        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        // Same distortion polynomial as ReprojectionCost / projectPointR
        T r2 = xp * xp + yp * yp;
        T r4 = r2 * r2;
        T r6 = r4 * r2;
        T radial = T(1) + T(dist[0]) * r2 + T(dist[1]) * r4 + T(dist[4]) * r6;
        T xpp = xp * radial + T(2) * T(dist[2]) * xp * yp +
                T(dist[3]) * (r2 + T(2) * xp * xp);
        T ypp = yp * radial + T(dist[2]) * (r2 + T(2) * yp * yp) +
                T(2) * T(dist[3]) * xp * yp;

        // intr = {f, cx, cy}; fy = f * fy_ratio
        T pred_x = intr[0] * xpp + intr[1];
        T pred_y = intr[0] * T(fy_ratio) * ypp + intr[2];

        residuals[0] = pred_x - T(obs_x);
        residuals[1] = pred_y - T(obs_y);
        return true;
    }

    static ceres::CostFunction *Create(const Eigen::Vector3d &X,
                                       const Eigen::Vector3d &rvec,
                                       const Eigen::Vector3d &tvec,
                                       const Eigen::Matrix<double, 5, 1> &dist,
                                       double fy_ratio,
                                       const Eigen::Vector2d &obs) {
        auto *c = new PostReprojectionCost();
        for (int i = 0; i < 3; i++) {
            c->X[i] = X(i);
            c->rvec[i] = rvec(i);
            c->tvec[i] = tvec(i);
        }
        for (int i = 0; i < 5; i++) c->dist[i] = dist(i);
        c->fy_ratio = fy_ratio;
        c->obs_x = obs.x();
        c->obs_y = obs.y();
        return new ceres::AutoDiffCostFunction<PostReprojectionCost, 2, 3>(c);
    }
};

// Prior pulling {f, cx, cy} toward the crop-shifted stage-1 values.
// Residuals are in px per px of deviation, directly comparable to
// reprojection residuals.
struct IntrinsicPriorCost {
    double f0, cx0, cy0;
    double wf, wc;

    template <typename T>
    bool operator()(const T *intr, T *residuals) const {
        residuals[0] = T(wf) * (intr[0] - T(f0));
        residuals[1] = T(wc) * (intr[1] - T(cx0));
        residuals[2] = T(wc) * (intr[2] - T(cy0));
        return true;
    }

    static ceres::CostFunction *Create(double f0, double cx0, double cy0,
                                       double wf, double wc) {
        auto *c = new IntrinsicPriorCost();
        c->f0 = f0;
        c->cx0 = cx0;
        c->cy0 = cy0;
        c->wf = wf;
        c->wc = wc;
        return new ceres::AutoDiffCostFunction<IntrinsicPriorCost, 3, 3>(c);
    }
};

// landmarks: cam serial -> post id -> pixel (OpenCV y-down, already flipped
// per camera by the caller).
inline RefineResult run_cropped_refinement(
    const RefineConfig &config,
    const std::map<std::string, std::map<int, Eigen::Vector2d>> &landmarks,
    std::string *status = nullptr) {

    RefineResult result;
    int nc = (int)config.camera_names.size();
    if (nc == 0) {
        result.error = "No cameras";
        return result;
    }

    // 1. Load cropped calibration YAMLs
    if (status) *status = "Loading calibration from " + config.calibration_folder;
    std::string calib_folder =
        CropCalibration::resolve_calibration_folder(config.calibration_folder);
    std::vector<CalibrationPipeline::CameraPose> poses(nc);
    std::vector<int> image_widths(nc, 0), image_heights(nc, 0);
    for (int c = 0; c < nc; c++) {
        std::string yaml_path =
            calib_folder + "/Cam" + config.camera_names[c] + ".yaml";
        if (!fs::exists(yaml_path)) {
            result.error = "Missing calibration file: " + yaml_path;
            return result;
        }
        try {
            auto yf = opencv_yaml::read(yaml_path);
            poses[c].K = yf.getMatrix("camera_matrix").block<3, 3>(0, 0);
            Eigen::MatrixXd dist_mat = yf.getMatrix("distortion_coefficients");
            for (int j = 0; j < 5; j++) poses[c].dist(j) = dist_mat(j, 0);
            poses[c].R = yf.getMatrix("rc_ext").block<3, 3>(0, 0);
            Eigen::MatrixXd t_mat = yf.getMatrix("tc_ext");
            poses[c].t = Eigen::Vector3d(t_mat(0, 0), t_mat(1, 0), t_mat(2, 0));
            image_widths[c] = yf.getInt("image_width");
            image_heights[c] = yf.getInt("image_height");
        } catch (const std::exception &e) {
            result.error = "Error reading " + yaml_path + ": " + e.what();
            return result;
        }
    }
    std::vector<CalibrationPipeline::CameraPose> original_poses = poses;

    // 2. Load fixed 3D posts
    auto posts_3d = CropCalibration::read_posts_3d_csv(config.posts_3d_csv);
    if ((int)posts_3d.size() < 3) {
        result.error = "Need at least 3 finite posts in " + config.posts_3d_csv +
                       " (found " + std::to_string(posts_3d.size()) + ")";
        return result;
    }

    // 3. Gather observations; drop gross outliers by INITIAL reprojection
    //    (bad clicks / swapped post ids).
    struct Obs {
        int cam;
        int post_id;
        Eigen::Vector2d px;
        double init_err;
    };
    std::vector<Obs> obs;
    std::set<int> observed_ids;
    for (int c = 0; c < nc; c++) {
        auto it = landmarks.find(config.camera_names[c]);
        if (it == landmarks.end()) continue;
        for (const auto &[pid, px] : it->second) {
            auto pit = posts_3d.find(pid);
            if (pit == posts_3d.end()) continue;
            auto pr = red_math::projectPointR(pit->second, poses[c].R,
                                              poses[c].t, poses[c].K,
                                              poses[c].dist);
            if (!pr.allFinite()) continue;
            double e = (pr - px).norm();
            if (e > config.outlier_th) {
                result.obs_dropped++;
                result.warnings.push_back(
                    "Dropped post " + std::to_string(pid) + " in Cam" +
                    config.camera_names[c] + ": initial reproj " +
                    std::to_string(e) + " px > threshold");
                continue;
            }
            obs.push_back({c, pid, px, e});
            observed_ids.insert(pid);
        }
    }
    if (obs.empty()) {
        result.error = "No usable post observations after outlier filtering";
        return result;
    }

    // 4. Holdout split by post id (deterministic)
    std::set<int> holdout_ids;
    if (config.holdout_fraction > 0 && config.holdout_fraction < 1.0) {
        std::vector<int> id_list(observed_ids.begin(), observed_ids.end());
        std::mt19937 rng(config.holdout_seed);
        std::shuffle(id_list.begin(), id_list.end(), rng);
        int n_hold = (int)std::lround(config.holdout_fraction * id_list.size());
        n_hold = std::min(n_hold, (int)id_list.size() - 3);  // keep >= 3 train posts
        for (int i = 0; i < n_hold; i++) holdout_ids.insert(id_list[i]);
    }

    // 5. Build per-camera parameter blocks {f, cx, cy} and the Ceres problem
    std::vector<std::array<double, 3>> intr(nc);
    std::vector<double> fy_ratio(nc);
    std::vector<int> train_count(nc, 0);
    for (int c = 0; c < nc; c++) {
        intr[c] = {poses[c].K(0, 0), poses[c].K(0, 2), poses[c].K(1, 2)};
        fy_ratio[c] = poses[c].K(1, 1) / poses[c].K(0, 0);
    }

    ceres::Problem problem;
    for (const auto &o : obs) {
        if (holdout_ids.count(o.post_id)) continue;
        auto rv = red_math::rotationMatrixToVector(poses[o.cam].R);
        problem.AddResidualBlock(
            PostReprojectionCost::Create(posts_3d.at(o.post_id), rv,
                                         poses[o.cam].t, poses[o.cam].dist,
                                         fy_ratio[o.cam], o.px),
            new ceres::HuberLoss(config.huber_delta), intr[o.cam].data());
        train_count[o.cam]++;
    }

    for (int c = 0; c < nc; c++) {
        if (train_count[c] == 0) {
            result.warnings.push_back("Cam" + config.camera_names[c] +
                                      ": no post observations — intrinsics "
                                      "left unchanged");
            continue;
        }
        int min_obs = config.free_focal ? 3 : 2;
        if (train_count[c] < min_obs)
            result.warnings.push_back(
                "Cam" + config.camera_names[c] + ": only " +
                std::to_string(train_count[c]) +
                " post observation(s) — refinement relies heavily on the prior");

        problem.AddResidualBlock(
            IntrinsicPriorCost::Create(intr[c][0], intr[c][1], intr[c][2],
                                       config.prior_focal_weight,
                                       config.prior_principal_weight),
            nullptr, intr[c].data());

        if (!config.free_focal) {
            // Principal-point-only mode: hold f (index 0) constant.
            problem.SetManifold(intr[c].data(),
                                new ceres::SubsetManifold(3, {0}));
        }
    }

    // 6. Solve — cameras are mutually independent, tiny problem
    if (status) *status = "Refining intrinsics...";
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_QR;
    options.max_num_iterations = config.max_iter;
    options.minimizer_progress_to_stdout = false;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (summary.termination_type == ceres::FAILURE) {
        result.error = "Ceres failed: " + summary.message;
        return result;
    }

    // 7. Unpack refined intrinsics
    for (int c = 0; c < nc; c++) {
        poses[c].K(0, 0) = intr[c][0];
        poses[c].K(1, 1) = intr[c][0] * fy_ratio[c];
        poses[c].K(0, 2) = intr[c][1];
        poses[c].K(1, 2) = intr[c][2];
    }

    // 8. Before/after statistics (direct projection of the fixed posts)
    auto reproj_err = [&](const CalibrationPipeline::CameraPose &pose,
                          const Obs &o) {
        auto pr = red_math::projectPointR(posts_3d.at(o.post_id), pose.R,
                                          pose.t, pose.K, pose.dist);
        return (pr - o.px).norm();
    };

    result.per_camera.resize(nc);
    std::vector<double> before_sum(nc, 0.0), after_sum(nc, 0.0);
    double tb = 0, ta = 0, hb = 0, ha = 0;
    int tn = 0, hn = 0;
    for (const auto &o : obs) {
        double eb = o.init_err;
        double ea = reproj_err(poses[o.cam], o);
        if (holdout_ids.count(o.post_id)) {
            hb += eb;
            ha += ea;
            hn++;
        } else {
            before_sum[o.cam] += eb;
            after_sum[o.cam] += ea;
            tb += eb;
            ta += ea;
            tn++;
        }
    }
    result.obs_used = tn;
    result.mean_before = tn ? tb / tn : 0.0;
    result.mean_after = tn ? ta / tn : 0.0;
    result.holdout_obs = hn;
    result.holdout_before = hn ? hb / hn : 0.0;
    result.holdout_after = hn ? ha / hn : 0.0;

    for (int c = 0; c < nc; c++) {
        auto &s = result.per_camera[c];
        s.serial = config.camera_names[c];
        s.n_obs = train_count[c];
        s.reproj_before = train_count[c] ? before_sum[c] / train_count[c] : 0.0;
        s.reproj_after = train_count[c] ? after_sum[c] / train_count[c] : 0.0;
        s.d_cx = poses[c].K(0, 2) - original_poses[c].K(0, 2);
        s.d_cy = poses[c].K(1, 2) - original_poses[c].K(1, 2);
        s.d_f = poses[c].K(0, 0) - original_poses[c].K(0, 0);
    }

    // 9. Write refined calibration + report
    if (status) *status = "Writing refined calibration...";
    std::string write_err;
    if (!CalibrationPipeline::write_calibration(
            poses, config.camera_names, config.output_folder, image_widths,
            image_heights, &write_err)) {
        result.error = write_err;
        return result;
    }

    // Carry crop provenance forward
    std::error_code ec;
    if (fs::exists(calib_folder + "/crop_info.json"))
        fs::copy_file(calib_folder + "/crop_info.json",
                      config.output_folder + "/crop_info.json",
                      fs::copy_options::overwrite_existing, ec);

    {
        nlohmann::json j;
        j["config"] = {{"calibration_folder", config.calibration_folder},
                       {"posts_3d_csv", config.posts_3d_csv},
                       {"free_focal", config.free_focal},
                       {"prior_principal_weight", config.prior_principal_weight},
                       {"prior_focal_weight", config.prior_focal_weight},
                       {"huber_delta", config.huber_delta},
                       {"outlier_th", config.outlier_th},
                       {"holdout_fraction", config.holdout_fraction},
                       {"holdout_seed", config.holdout_seed}};
        j["mean_before"] = result.mean_before;
        j["mean_after"] = result.mean_after;
        j["holdout_before"] = result.holdout_before;
        j["holdout_after"] = result.holdout_after;
        j["holdout_obs"] = result.holdout_obs;
        j["obs_used"] = result.obs_used;
        j["obs_dropped"] = result.obs_dropped;
        j["warnings"] = result.warnings;
        nlohmann::json cams = nlohmann::json::array();
        for (const auto &s : result.per_camera)
            cams.push_back({{"serial", s.serial},
                            {"n_obs", s.n_obs},
                            {"reproj_before", s.reproj_before},
                            {"reproj_after", s.reproj_after},
                            {"d_cx", s.d_cx},
                            {"d_cy", s.d_cy},
                            {"d_f", s.d_f}});
        j["per_camera"] = cams;
        std::ofstream f(config.output_folder + "/refine_report.json");
        if (f.is_open()) f << j.dump(2) << "\n";
    }

    result.success = true;
    result.output_folder = config.output_folder;
    if (status)
        *status = "Refinement complete: " + std::to_string(result.mean_before)
                  .substr(0, 5) + " -> " +
                  std::to_string(result.mean_after).substr(0, 5) + " px";
    return result;
}

}  // namespace CroppedRefinement
