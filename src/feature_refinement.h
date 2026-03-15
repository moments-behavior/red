#pragma once
#include "calibration_pipeline.h" // ReprojectionCost, CameraPose, write_calibration, triangulate_landmarks_multiview
#include "calibration_tool.h"     // CalibConfig (for cam_ordered)
#include "opencv_yaml_io.h"
#include "red_math.h"
#include "json.hpp"
#include <ceres/ceres.h>
#include <filesystem>
#include <fstream>
#include <chrono>
#include <thread>
#include <unordered_map>

namespace FeatureRefinement {
namespace fs = std::filesystem;

struct FeatureConfig {
    std::string landmarks_file;
    std::string points_3d_file;      // optional: pre-triangulated 3D points from Python
    std::string calibration_folder;
    std::string output_folder;
    std::vector<std::string> camera_names;

    double reproj_threshold = 8.0;

    double ba_outlier_th1 = 10.0;
    double ba_outlier_th2 = 3.0;
    int ba_max_iter = 50;
    bool lock_intrinsics = true;
    bool lock_distortion = true;
    double prior_rot_weight = 10.0;    // rotation prior (allow ~6° adjustment)
    double prior_trans_weight = 100.0; // translation prior (mounts don't slide)

    // Multi-round convergence
    int ba_max_rounds = 5;             // max re-triangulate+BA rounds
    double ba_convergence_eps = 0.001; // stop when reproj change < this (px)
};

struct FeatureResult {
    bool success = false;
    std::string error;
    int total_tracks = 0;
    int valid_3d_points = 0;
    int total_observations = 0;
    int ba_outliers_removed = 0;
    double mean_reproj_before = 0.0;
    double mean_reproj_after = 0.0;
    std::string output_folder;

    struct CameraChange {
        std::string name;
        double d_rot_deg = 0;
        double d_trans_mm = 0;
        double d_fx = 0, d_fy = 0, d_cx = 0, d_cy = 0;
    };
    std::vector<CameraChange> camera_changes;

    // Multi-round convergence diagnostics
    int ba_rounds_completed = 0;
    std::vector<double> per_round_reproj;
};

// ---------------------------------------------------------------------------
// Pose prior cost: penalizes deviation from initial camera parameters
// ---------------------------------------------------------------------------
struct PosePriorCost {
    double init_params[6]; // initial rvec(3) + tvec(3)
    double rot_weight;     // weight for rotation (rvec) deviations
    double trans_weight;   // weight for translation deviations

    PosePriorCost(const double *init, double rw, double tw)
        : rot_weight(rw), trans_weight(tw) {
        for (int i = 0; i < 6; i++) init_params[i] = init[i];
    }

    template <typename T>
    bool operator()(const T *const camera, T *residuals) const {
        // Rotation prior (rvec, indices 0-2): cameras on rigid mounts rotate more easily
        for (int i = 0; i < 3; i++)
            residuals[i] = T(rot_weight) * (camera[i] - T(init_params[i]));
        // Translation prior (tvec, indices 3-5): mounts don't slide, stronger constraint
        for (int i = 3; i < 6; i++)
            residuals[i] = T(trans_weight) * (camera[i] - T(init_params[i]));
        return true;
    }

    static ceres::CostFunction *Create(const double *init, double rot_weight, double trans_weight) {
        return new ceres::AutoDiffCostFunction<PosePriorCost, 6, 15>(
            new PosePriorCost(init, rot_weight, trans_weight));
    }
};

// ---------------------------------------------------------------------------
// Bundle adjustment with configurable intrinsic/distortion locking
// ---------------------------------------------------------------------------
inline double bundle_adjust_features(
    int nc,
    const std::vector<std::string> &cam_names,
    const std::map<std::string, std::map<int, Eigen::Vector2d>> &landmarks,
    std::vector<CalibrationPipeline::CameraPose> &poses,
    std::map<int, Eigen::Vector3d> &points_3d,
    double outlier_th, int max_iter, bool lock_intrinsics, bool lock_distortion,
    double prior_rot_weight, double prior_trans_weight,
    int &outliers_removed) {

    // Pack cameras: 15 params each [rvec(3), tvec(3), fx, fy, cx, cy, k1, k2, p1, p2, k3]
    // Handle improper rotations (det(R)=-1) by negating both R and t.
    // This preserves the projection: (-R)*X + (-t) = -(R*X + t), and the
    // perspective division x/z is unaffected by the overall sign.
    std::vector<bool> r_negated(nc, false);
    std::vector<std::array<double, 15>> cp(nc);
    for (int i = 0; i < nc; i++) {
        Eigen::Matrix3d R = poses[i].R;
        Eigen::Vector3d t = poses[i].t;
        if (R.determinant() < 0) { R = -R; t = -t; r_negated[i] = true; }
        auto rv = red_math::rotationMatrixToVector(R);
        cp[i] = {rv.x(), rv.y(), rv.z(), t.x(), t.y(), t.z(),
                 poses[i].K(0,0), poses[i].K(1,1), poses[i].K(0,2), poses[i].K(1,2),
                 poses[i].dist(0), poses[i].dist(1), poses[i].dist(2), poses[i].dist(3), poses[i].dist(4)};
    }

    // Pack 3D points
    std::vector<int> pids;
    for (const auto &[id, _] : points_3d) pids.push_back(id);
    std::sort(pids.begin(), pids.end());
    std::unordered_map<int, int> pidx;
    std::vector<std::array<double, 3>> pp(pids.size());
    for (int i = 0; i < (int)pids.size(); i++) {
        pidx[pids[i]] = i;
        const auto &pt = points_3d[pids[i]];
        pp[i] = {pt.x(), pt.y(), pt.z()};
    }

    // Build observations
    struct Obs { int ci, pi; double px, py; };
    std::vector<Obs> obs;
    for (int c = 0; c < nc; c++) {
        auto it = landmarks.find(cam_names[c]);
        if (it == landmarks.end()) continue;
        for (const auto &[pid, px] : it->second) {
            auto pit = pidx.find(pid);
            if (pit != pidx.end()) obs.push_back({c, pit->second, px.x(), px.y()});
        }
    }

    // Build Ceres problem
    ceres::Problem problem;
    for (const auto &o : obs) {
        problem.AddResidualBlock(
            CalibrationPipeline::ReprojectionCost::Create(o.px, o.py),
            new ceres::HuberLoss(1.0),
            cp[o.ci].data(), pp[o.pi].data());
    }

    // Find anchor camera (most observations) before adding priors
    std::map<int, int> cam_obs_count;
    for (const auto &o : obs) cam_obs_count[o.ci]++;
    int anchor_cam = -1;
    int max_obs = 0;
    for (const auto &[ci, count] : cam_obs_count) {
        if (count > max_obs) { max_obs = count; anchor_cam = ci; }
    }
    if (anchor_cam < 0) return 0.0;  // no observations at all

    // Pose prior: penalize deviations from initial calibration.
    // Skip anchor (its extrinsics are fixed, prior would be wasted work).
    for (int c = 0; c < nc; c++) {
        if (c == anchor_cam) continue;
        problem.AddResidualBlock(
            PosePriorCost::Create(cp[c].data(), prior_rot_weight, prior_trans_weight),
            nullptr,  // no robust loss on prior
            cp[c].data());
    }

    // Set manifolds for all active cameras
    for (const auto &[ci, count] : cam_obs_count) {
        std::vector<int> fixed;
        if (ci == anchor_cam) {
            // Anchor: fix extrinsics (gauge constraint)
            fixed = {0, 1, 2, 3, 4, 5};
        }
        if (lock_intrinsics) { fixed.push_back(6); fixed.push_back(7); fixed.push_back(8); fixed.push_back(9); }
        if (lock_distortion) { fixed.push_back(10); fixed.push_back(11); fixed.push_back(12); fixed.push_back(13); fixed.push_back(14); }
        if (!fixed.empty())
            problem.SetManifold(cp[ci].data(), new ceres::SubsetManifold(15, fixed));
    }

    ceres::Solver::Options options;
    options.linear_solver_type = ceres::SPARSE_SCHUR;
    options.max_num_iterations = max_iter;
    options.num_threads = std::max(1, (int)std::thread::hardware_concurrency());
    options.minimizer_progress_to_stdout = false;

    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    // Precompute per-camera projection params from packed array
    struct CamProj { Eigen::Vector3d rv, tv; Eigen::Matrix3d K; Eigen::Matrix<double, 5, 1> d; };
    std::vector<CamProj> cam_proj(nc);
    for (int i = 0; i < nc; i++) {
        cam_proj[i].rv = Eigen::Vector3d(cp[i][0], cp[i][1], cp[i][2]);
        cam_proj[i].tv = Eigen::Vector3d(cp[i][3], cp[i][4], cp[i][5]);
        cam_proj[i].K = Eigen::Matrix3d::Identity();
        cam_proj[i].K(0,0) = cp[i][6]; cam_proj[i].K(1,1) = cp[i][7];
        cam_proj[i].K(0,2) = cp[i][8]; cam_proj[i].K(1,2) = cp[i][9];
        cam_proj[i].d << cp[i][10], cp[i][11], cp[i][12], cp[i][13], cp[i][14];
    }

    // Compute reprojection errors and reject outliers
    outliers_removed = 0;
    std::set<int> bad_points;
    double err_sum = 0;
    int err_count = 0;
    for (const auto &o : obs) {
        const auto &cp_i = cam_proj[o.ci];
        auto pr = red_math::projectPoint(Eigen::Vector3d(pp[o.pi][0], pp[o.pi][1], pp[o.pi][2]),
                                          cp_i.rv, cp_i.tv, cp_i.K, cp_i.d);
        double e = (pr - Eigen::Vector2d(o.px, o.py)).norm();
        if (e > outlier_th) { bad_points.insert(pids[o.pi]); outliers_removed++; }
        else { err_sum += e; err_count++; }
    }
    for (int id : bad_points) points_3d.erase(id);

    // Unpack cameras (restore improper rotation sign if needed)
    for (int i = 0; i < nc; i++) {
        Eigen::Vector3d rv(cp[i][0], cp[i][1], cp[i][2]);
        poses[i].R = red_math::rotationVectorToMatrix(rv);
        poses[i].t = Eigen::Vector3d(cp[i][3], cp[i][4], cp[i][5]);
        if (r_negated[i]) { poses[i].R = -poses[i].R; poses[i].t = -poses[i].t; }
        poses[i].K = Eigen::Matrix3d::Identity();
        poses[i].K(0,0) = cp[i][6]; poses[i].K(1,1) = cp[i][7]; poses[i].K(0,2) = cp[i][8]; poses[i].K(1,2) = cp[i][9];
        poses[i].dist << cp[i][10], cp[i][11], cp[i][12], cp[i][13], cp[i][14];
    }
    // Unpack surviving points
    for (int i = 0; i < (int)pids.size(); i++) {
        if (points_3d.count(pids[i]))
            points_3d[pids[i]] = Eigen::Vector3d(pp[i][0], pp[i][1], pp[i][2]);
    }

    return err_count > 0 ? err_sum / err_count : 0.0;
}

// ---------------------------------------------------------------------------
// Main entry point
// ---------------------------------------------------------------------------
inline FeatureResult run_feature_refinement(const FeatureConfig &config, std::string *status = nullptr) {
    FeatureResult result;
    auto t0 = std::chrono::steady_clock::now();
    int nc = (int)config.camera_names.size();

    // 1. Load calibration YAMLs
    if (status) *status = "Loading calibration from " + config.calibration_folder;
    std::vector<CalibrationPipeline::CameraPose> poses(nc);
    int image_width = 0, image_height = 0;

    for (int c = 0; c < nc; c++) {
        std::string yaml_path = config.calibration_folder + "/Cam" + config.camera_names[c] + ".yaml";
        if (!fs::exists(yaml_path)) {
            result.error = "Missing calibration file: " + yaml_path;
            return result;
        }
        try {
            auto yaml = opencv_yaml::read(yaml_path);
            poses[c].K = yaml.getMatrix("camera_matrix").block<3, 3>(0, 0);
            Eigen::MatrixXd dist_mat = yaml.getMatrix("distortion_coefficients");
            for (int j = 0; j < 5; j++) poses[c].dist(j) = dist_mat(j, 0);
            poses[c].R = yaml.getMatrix("rc_ext").block<3, 3>(0, 0);
            Eigen::MatrixXd t_mat = yaml.getMatrix("tc_ext");
            poses[c].t = Eigen::Vector3d(t_mat(0, 0), t_mat(1, 0), t_mat(2, 0));
            if (c == 0) { image_width = yaml.getInt("image_width"); image_height = yaml.getInt("image_height"); }
            fprintf(stderr, "Feature: loaded %s (fx=%.1f fy=%.1f)\n", config.camera_names[c].c_str(), poses[c].K(0,0), poses[c].K(1,1));
        } catch (const std::exception &e) {
            result.error = "Error reading " + yaml_path + ": " + e.what();
            return result;
        }
    }

    // Save original poses for delta computation
    std::vector<CalibrationPipeline::CameraPose> original_poses = poses;

    // 2. Read landmarks.json
    if (status) *status = "Loading landmarks from " + config.landmarks_file;
    std::map<std::string, std::map<int, Eigen::Vector2d>> landmarks;
    try {
        std::ifstream f(config.landmarks_file);
        if (!f.is_open()) { result.error = "Cannot open " + config.landmarks_file; return result; }
        nlohmann::json j;
        f >> j;
        for (auto &[cam, cam_j] : j.items()) {
            auto &ids = cam_j["ids"];
            auto &pts = cam_j["landmarks"];
            for (int i = 0; i < (int)ids.size(); i++) {
                landmarks[cam][ids[i].get<int>()] = Eigen::Vector2d(pts[i][0].get<double>(), pts[i][1].get<double>());
            }
        }
    } catch (const std::exception &e) {
        result.error = std::string("Error parsing landmarks: ") + e.what();
        return result;
    }

    // Count total tracks (unique point IDs across all cameras)
    std::set<int> all_ids;
    int total_obs = 0;
    for (const auto &[cam, pts] : landmarks) {
        total_obs += (int)pts.size();
        for (const auto &[id, _] : pts) all_ids.insert(id);
    }
    result.total_tracks = (int)all_ids.size();
    result.total_observations = total_obs;
    fprintf(stderr, "Feature: %d tracks, %d observations across %d cameras\n", result.total_tracks, total_obs, nc);

    // 3. Create minimal CalibConfig for triangulation
    CalibrationTool::CalibConfig calib_config;
    calib_config.cam_ordered = config.camera_names;

    // 4. Triangulate
    if (status) *status = "Triangulating " + std::to_string(result.total_tracks) + " tracks...";
    // 4. Load or triangulate 3D points
    std::map<int, Eigen::Vector3d> points_3d;
    if (!config.points_3d_file.empty() && fs::exists(config.points_3d_file)) {
        // Load pre-triangulated points from Python (preferred)
        if (status) *status = "Loading 3D points from " + config.points_3d_file;
        try {
            std::ifstream pf(config.points_3d_file);
            nlohmann::json pj;
            pf >> pj;
            for (auto &[id_str, pt] : pj.items()) {
                points_3d[std::stoi(id_str)] = Eigen::Vector3d(
                    pt[0].get<double>(), pt[1].get<double>(), pt[2].get<double>());
            }
        } catch (const std::exception &e) {
            result.error = std::string("Error parsing points_3d: ") + e.what();
            return result;
        }
        fprintf(stderr, "Feature: loaded %d pre-triangulated 3D points\n", (int)points_3d.size());
    } else {
        // Triangulate from landmarks using calibration
        if (status) *status = "Triangulating " + std::to_string(result.total_tracks) + " tracks...";
        CalibrationPipeline::triangulate_landmarks_multiview(
            calib_config, landmarks, poses, points_3d, config.reproj_threshold);
        fprintf(stderr, "Feature: triangulated %d / %d tracks (threshold=%.0f)\n",
                (int)points_3d.size(), result.total_tracks, config.reproj_threshold);
    }
    int np = (int)points_3d.size();
    result.valid_3d_points = np;

    if (np < 10) {
        result.error = "Too few valid 3D points (" + std::to_string(np) + "), need at least 10";
        return result;
    }

    // Compute initial reprojection error (before any BA)
    {
        double err_sum = 0; int err_count = 0;
        for (int c = 0; c < nc; c++) {
            auto it = landmarks.find(config.camera_names[c]);
            if (it == landmarks.end()) continue;
            for (const auto &[pid, px] : it->second) {
                auto pit = points_3d.find(pid);
                if (pit == points_3d.end()) continue;
                auto pr = red_math::projectPointR(pit->second, poses[c].R, poses[c].t, poses[c].K, poses[c].dist);
                err_sum += (pr - px).norm();
                err_count++;
            }
        }
        result.mean_reproj_before = err_count > 0 ? err_sum / err_count : 0.0;
        fprintf(stderr, "Feature: initial reproj error: %.3f px (%d observations)\n",
                result.mean_reproj_before, err_count);
    }

    // 5. Multi-round BA with re-triangulation for convergence
    //    Modeled on bundle_adjust_experimental in calibration_pipeline.h.
    //    Each round: re-triangulate with updated extrinsics, then BA with
    //    progressively tighter outlier threshold. Stops when reproj error
    //    change < convergence_eps or max_rounds reached.

    int total_outliers = 0;

    // Coarse pass: large outlier threshold to remove gross errors
    if (status) *status = "BA coarse pass (" + std::to_string(np) + " points)...";
    int outliers_coarse = 0;
    double prev_reproj = bundle_adjust_features(
        nc, config.camera_names, landmarks, poses, points_3d,
        config.ba_outlier_th1, config.ba_max_iter, config.lock_intrinsics, config.lock_distortion,
        config.prior_rot_weight, config.prior_trans_weight, outliers_coarse);
    total_outliers += outliers_coarse;
    result.per_round_reproj.push_back(prev_reproj);
    fprintf(stderr, "Feature: BA coarse — %.3f px, %d outliers, %d points\n",
            prev_reproj, outliers_coarse, (int)points_3d.size());

    // Convergence loop: re-triangulate + BA with tightening threshold
    int round = 0;
    for (; round < config.ba_max_rounds; round++) {
        // Re-triangulate all tracks with updated extrinsics
        std::map<int, Eigen::Vector3d> points_3d_new;
        CalibrationPipeline::triangulate_landmarks_multiview(
            calib_config, landmarks, poses, points_3d_new, config.reproj_threshold);
        // Merge: add newly triangulated points (keep existing ones as-is since BA optimized them)
        int new_pts = 0;
        for (const auto &[id, pt] : points_3d_new) {
            if (!points_3d.count(id)) { points_3d[id] = pt; new_pts++; }
        }

        // Progressive outlier threshold: 10 -> 8 -> 6 -> 4 -> 3 (floor)
        double outlier_th = std::max(config.ba_outlier_th2,
                                      config.ba_outlier_th1 - (round + 1) * 2.0);

        if (status) *status = "BA round " + std::to_string(round + 1) + "/" +
            std::to_string(config.ba_max_rounds) + " (th=" +
            std::to_string(outlier_th).substr(0, 4) + " px, " +
            std::to_string((int)points_3d.size()) + " pts)...";

        int outliers_round = 0;
        double reproj = bundle_adjust_features(
            nc, config.camera_names, landmarks, poses, points_3d,
            outlier_th, config.ba_max_iter, config.lock_intrinsics, config.lock_distortion,
            config.prior_rot_weight, config.prior_trans_weight, outliers_round);
        total_outliers += outliers_round;
        result.per_round_reproj.push_back(reproj);

        fprintf(stderr, "Feature: BA round %d — %.3f px (delta=%.4f), th=%.1f, +%d new, -%d outliers, %d pts\n",
                round + 1, reproj, std::abs(prev_reproj - reproj), outlier_th,
                new_pts, outliers_round, (int)points_3d.size());

        // Check convergence
        if (std::abs(prev_reproj - reproj) < config.ba_convergence_eps) {
            fprintf(stderr, "Feature: converged after %d rounds (delta < %.4f px)\n",
                    round + 1, config.ba_convergence_eps);
            round++;
            break;
        }
        prev_reproj = reproj;
    }

    // Final tight pass
    if (status) *status = "BA final pass (th=" + std::to_string(config.ba_outlier_th2).substr(0, 4) + " px)...";
    int outliers_final = 0;
    result.mean_reproj_after = bundle_adjust_features(
        nc, config.camera_names, landmarks, poses, points_3d,
        config.ba_outlier_th2, config.ba_max_iter, config.lock_intrinsics, config.lock_distortion,
        config.prior_rot_weight, config.prior_trans_weight, outliers_final);
    total_outliers += outliers_final;
    result.per_round_reproj.push_back(result.mean_reproj_after);
    result.ba_outliers_removed = total_outliers;
    result.ba_rounds_completed = round + 2; // coarse + rounds + final
    result.valid_3d_points = (int)points_3d.size();
    fprintf(stderr, "Feature: BA final — %.3f px, %d outliers, %d points (%d total rounds)\n",
            result.mean_reproj_after, outliers_final, (int)points_3d.size(), result.ba_rounds_completed);

    // 7. Compute per-camera changes
    result.camera_changes.resize(nc);
    for (int c = 0; c < nc; c++) {
        auto &ch = result.camera_changes[c];
        ch.name = config.camera_names[c];
        Eigen::Matrix3d dR = original_poses[c].R.transpose() * poses[c].R;
        double trace_val = std::min(3.0, std::max(-1.0, dR.trace()));
        ch.d_rot_deg = std::acos((trace_val - 1.0) / 2.0) * 180.0 / M_PI;
        ch.d_trans_mm = (poses[c].t - original_poses[c].t).norm();
        ch.d_fx = poses[c].K(0,0) - original_poses[c].K(0,0);
        ch.d_fy = poses[c].K(1,1) - original_poses[c].K(1,1);
        ch.d_cx = poses[c].K(0,2) - original_poses[c].K(0,2);
        ch.d_cy = poses[c].K(1,2) - original_poses[c].K(1,2);
    }

    // Print summary table
    fprintf(stderr, "\n%-12s %8s %10s %6s %6s %6s %6s\n", "Camera", "Rot(deg)", "Trans(mm)", "dFx", "dFy", "dCx", "dCy");
    fprintf(stderr, "%-12s %8s %10s %6s %6s %6s %6s\n", "------", "--------", "--------", "----", "----", "----", "----");
    for (const auto &ch : result.camera_changes)
        fprintf(stderr, "%-12s %8.4f %10.4f %6.2f %6.2f %6.2f %6.2f\n",
                ch.name.c_str(), ch.d_rot_deg, ch.d_trans_mm, ch.d_fx, ch.d_fy, ch.d_cx, ch.d_cy);

    // 8. Write refined YAMLs
    std::string out_folder = config.output_folder;
    if (out_folder.empty())
        out_folder = config.calibration_folder + "/feature_refined";
    if (status) *status = "Writing refined calibration to " + out_folder;
    std::string write_err;
    if (!CalibrationPipeline::write_calibration(poses, config.camera_names, out_folder, image_width, image_height, &write_err)) {
        result.error = write_err;
        return result;
    }
    result.output_folder = out_folder;

    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    result.success = true;
    fprintf(stderr, "\nFeature refinement done in %.1f s: %.3f px -> %.3f px, %d points, %d outliers, output: %s\n",
            elapsed, result.mean_reproj_before, result.mean_reproj_after, result.valid_3d_points, result.ba_outliers_removed, out_folder.c_str());
    if (status) *status = "Feature refinement done: " + std::to_string(result.mean_reproj_after).substr(0, 5) +
        " px (" + std::to_string(result.valid_3d_points) + " points, " + std::to_string(result.ba_outliers_removed) + " outliers)";

    return result;
}

} // namespace FeatureRefinement
