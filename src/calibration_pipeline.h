#pragma once
// calibration_pipeline.h — Full multiview calibration pipeline (C++).
// Ports the multiview_calib Python pipeline: ChArUco detection → intrinsics →
// pairwise relative poses → chained global poses → bundle adjustment →
// world registration → per-camera YAML output.

#include "calibration_tool.h"
#include "opencv_yaml_io.h"
#include "red_math.h"

#include "../lib/ImGuiFileDialog/stb/stb_image.h"

#include <opencv2/aruco/aruco_calib.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect/aruco_board.hpp>
#include <opencv2/objdetect/charuco_detector.hpp>

#include <ceres/ceres.h>
#include <ceres/rotation.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <Eigen/SVD>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <map>
#include <mutex>
#include <numeric>
#include <set>
#include <string>
#include <vector>

namespace CalibrationPipeline {

// ─────────────────────────────────────────────────────────────────────────────
// Data structures
// ─────────────────────────────────────────────────────────────────────────────

struct CameraIntrinsics {
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 5, 1> dist = Eigen::Matrix<double, 5, 1>::Zero();
    double reproj_error = 0.0;
    int image_width = 0;
    int image_height = 0;
    // Per-image detected corners: image_index → {corner_id → pixel}
    std::map<int, std::vector<cv::Point2f>> corners_per_image;
    std::map<int, std::vector<int>> ids_per_image;
};

struct RelativePose {
    Eigen::Matrix3d Rd = Eigen::Matrix3d::Identity();
    Eigen::Vector3d td = Eigen::Vector3d::Zero();
    std::vector<Eigen::Vector3d> triang_points; // 3D points in cam_a's frame
    std::vector<int> point_ids;                 // global landmark IDs
};

struct CameraPose {
    Eigen::Matrix3d R = Eigen::Matrix3d::Identity();
    Eigen::Vector3d t = Eigen::Vector3d::Zero();
    Eigen::Matrix3d K = Eigen::Matrix3d::Identity();
    Eigen::Matrix<double, 5, 1> dist = Eigen::Matrix<double, 5, 1>::Zero();
};

struct CalibrationResult {
    std::vector<CameraPose> cameras;
    std::vector<std::string> cam_names;
    std::map<int, Eigen::Vector3d> points_3d; // global_id → world point
    double mean_reproj_error = 0.0;
    int image_width = 0;
    int image_height = 0;
    bool success = false;
    std::string error;
    std::string output_folder; // timestamped folder where YAMLs were written
};

// ─────────────────────────────────────────────────────────────────────────────
// Helpers
// ─────────────────────────────────────────────────────────────────────────────

// Get sorted list of image numbers for a camera serial.
inline std::vector<int>
get_sorted_image_numbers(const std::string &img_path,
                         const std::string &serial) {
    namespace fs = std::filesystem;
    std::vector<int> numbers;
    std::string prefix = serial + "_";
    for (const auto &entry : fs::directory_iterator(img_path)) {
        if (!entry.is_regular_file())
            continue;
        std::string fn = entry.path().filename().string();
        std::string ext = entry.path().extension().string();
        if (ext != ".jpg" && ext != ".jpeg" && ext != ".png" &&
            ext != ".tiff" && ext != ".tif")
            continue;
        if (fn.substr(0, prefix.size()) != prefix)
            continue;
        // Extract number: everything between prefix and extension
        std::string num_str =
            fn.substr(prefix.size(), fn.size() - prefix.size() - ext.size());
        try {
            numbers.push_back(std::stoi(num_str));
        } catch (...) {
            continue;
        }
    }
    std::sort(numbers.begin(), numbers.end());
    return numbers;
}

// Get image file extension for a camera serial (assumes consistent extension).
inline std::string get_image_extension(const std::string &img_path,
                                       const std::string &serial) {
    namespace fs = std::filesystem;
    std::string prefix = serial + "_";
    for (const auto &entry : fs::directory_iterator(img_path)) {
        if (!entry.is_regular_file())
            continue;
        std::string fn = entry.path().filename().string();
        if (fn.substr(0, prefix.size()) == prefix)
            return entry.path().extension().string();
    }
    return ".jpg";
}

// Convert OpenCV aruco dictionary ID to predefined dictionary.
inline cv::aruco::PredefinedDictionaryType
aruco_dict_from_id(int id) {
    return static_cast<cv::aruco::PredefinedDictionaryType>(id);
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 1: Detect ChArUco corners and calibrate intrinsics (per camera)
// ─────────────────────────────────────────────────────────────────────────────

inline bool
detect_and_calibrate_intrinsics(
    const CalibrationTool::CalibConfig &config,
    std::map<std::string, CameraIntrinsics> &intrinsics,
    std::string *status) {

    const auto &cs = config.charuco_setup;
    int max_corners = (cs.w - 1) * (cs.h - 1);

    // Get sorted image numbers (should be same across cameras).
    // Use the first camera to determine the image list, then verify others.
    auto image_numbers =
        get_sorted_image_numbers(config.img_path, config.cam_ordered[0]);
    if (image_numbers.empty()) {
        if (status)
            *status = "Error: No images found for " + config.cam_ordered[0];
        return false;
    }

    // Build image_number → sorted_index map
    std::map<int, int> img_num_to_idx;
    for (int i = 0; i < (int)image_numbers.size(); i++)
        img_num_to_idx[image_numbers[i]] = i;

    int num_cameras = (int)config.cam_ordered.size();
    std::mutex status_mutex;
    std::atomic<int> cameras_done{0};

    // Per-camera detection results (filled in parallel)
    struct DetectionResult {
        CameraIntrinsics cam_data; // corners_per_image, ids_per_image, image dims
        std::vector<std::vector<cv::Point3f>> all_obj_points;
        std::vector<std::vector<cv::Point2f>> all_img_points;
        cv::Size image_size;
        bool ok = false;
        std::string error;
    };
    std::vector<DetectionResult> detections(num_cameras);

    // ── Phase 1: detect ChArUco corners in parallel (thread-safe) ──
    if (status) *status = "Detecting ChArUco corners...";
    {
        std::vector<std::future<void>> futures;
        for (int cam_i = 0; cam_i < num_cameras; cam_i++) {
            futures.push_back(std::async(std::launch::async, [&, cam_i]() {
                const std::string &serial = config.cam_ordered[cam_i];
                std::string ext = get_image_extension(config.img_path, serial);
                auto &det = detections[cam_i];

                auto dictionary = cv::aruco::getPredefinedDictionary(
                    aruco_dict_from_id(cs.dictionary));
                auto board = cv::makePtr<cv::aruco::CharucoBoard>(
                    cv::Size(cs.w, cs.h), cs.square_side_length,
                    cs.marker_side_length, dictionary);
                cv::aruco::CharucoDetector detector(*board);

                for (int img_num : image_numbers) {
                    std::string img_file = config.img_path + "/" + serial +
                                           "_" + std::to_string(img_num) + ext;
                    int w = 0, h = 0, channels = 0;
                    unsigned char *pixels =
                        stbi_load(img_file.c_str(), &w, &h, &channels, 1);
                    if (!pixels)
                        continue;
                    cv::Mat img(h, w, CV_8UC1, pixels);

                    if (det.image_size.width == 0) {
                        det.image_size = cv::Size(w, h);
                        det.cam_data.image_width = w;
                        det.cam_data.image_height = h;
                    }

                    std::vector<cv::Point2f> corners;
                    std::vector<int> ids;
                    detector.detectBoard(img, corners, ids);

                    if ((int)ids.size() < 6) {
                        stbi_image_free(pixels);
                        continue;
                    }

                    cv::cornerSubPix(
                        img, corners, cv::Size(3, 3), cv::Size(-1, -1),
                        cv::TermCriteria(cv::TermCriteria::EPS +
                                             cv::TermCriteria::COUNT,
                                         30, 0.01));

                    stbi_image_free(pixels);

                    int sorted_idx = img_num_to_idx[img_num];
                    det.cam_data.corners_per_image[sorted_idx] = corners;
                    det.cam_data.ids_per_image[sorted_idx] =
                        std::vector<int>(ids.begin(), ids.end());

                    std::vector<cv::Point3f> obj_pts;
                    std::vector<cv::Point2f> img_pts;
                    board->matchImagePoints(corners, ids, obj_pts, img_pts);

                    if ((int)obj_pts.size() >= 6) {
                        det.all_obj_points.push_back(obj_pts);
                        det.all_img_points.push_back(img_pts);
                    }
                }

                if (det.all_obj_points.size() < 4) {
                    det.error = "Too few valid images for camera " + serial +
                                " (" +
                                std::to_string(det.all_obj_points.size()) + ")";
                } else {
                    det.ok = true;
                }

                int done = ++cameras_done;
                {
                    std::lock_guard<std::mutex> lock(status_mutex);
                    if (status)
                        *status = "Detecting corners (" + std::to_string(done) +
                                  "/" + std::to_string(num_cameras) + ")...";
                }
            }));
        }
        for (auto &f : futures)
            f.get();
    }

    // Check detection results
    for (int cam_i = 0; cam_i < num_cameras; cam_i++) {
        if (!detections[cam_i].ok) {
            if (status) *status = "Error: " + detections[cam_i].error;
            return false;
        }
    }

    // ── Phase 2: calibrate intrinsics serially (LAPACK not thread-safe) ──
    std::vector<CameraIntrinsics> results(num_cameras);
    std::vector<bool> result_ok(num_cameras, false);

    for (int cam_i = 0; cam_i < num_cameras; cam_i++) {
        auto &det = detections[cam_i];
        const std::string &serial = config.cam_ordered[cam_i];

        if (status)
            *status = "Calibrating intrinsics (" + std::to_string(cam_i + 1) +
                      "/" + std::to_string(num_cameras) + "): " + serial;

        cv::Mat K, dist_coeffs, rvecs, tvecs;
        int calib_flags = cv::CALIB_FIX_ASPECT_RATIO;
        double reproj_err = cv::calibrateCamera(
            det.all_obj_points, det.all_img_points, det.image_size, K,
            dist_coeffs, rvecs, tvecs, calib_flags);

        cv::cv2eigen(K, det.cam_data.K);

        det.cam_data.dist.setZero();
        int n = std::min((int)(dist_coeffs.total()), 5);
        for (int i = 0; i < n; i++)
            det.cam_data.dist(i) = dist_coeffs.at<double>(i);

        det.cam_data.reproj_error = reproj_err;
        results[cam_i] = std::move(det.cam_data);
        result_ok[cam_i] = true;
    }

    // Collect results
    for (int i = 0; i < num_cameras; i++) {
        if (!result_ok[i]) {
            if (status)
                *status = "Error: calibration failed for camera " +
                          config.cam_ordered[i];
            return false;
        }
        intrinsics[config.cam_ordered[i]] = std::move(results[i]);
    }

    if (status) {
        std::string msg = "Intrinsics done. Reproj errors:";
        for (const auto &[name, intr] : intrinsics)
            msg += " " + name + "=" +
                   std::to_string(intr.reproj_error).substr(0, 5);
        *status = msg;
    }
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 2: Build unified landmark map from per-camera detections
// ─────────────────────────────────────────────────────────────────────────────

inline void build_landmarks(
    const CalibrationTool::CalibConfig &config,
    const std::map<std::string, CameraIntrinsics> &intrinsics,
    std::map<std::string, std::map<int, Eigen::Vector2d>> &landmarks) {

    int max_corners =
        (config.charuco_setup.w - 1) * (config.charuco_setup.h - 1);

    for (const auto &serial : config.cam_ordered) {
        auto it = intrinsics.find(serial);
        if (it == intrinsics.end())
            continue;
        const auto &intr = it->second;
        auto &cam_landmarks = landmarks[serial];

        for (const auto &[img_idx, corners] : intr.corners_per_image) {
            const auto &ids = intr.ids_per_image.at(img_idx);
            for (int j = 0; j < (int)ids.size(); j++) {
                int global_id = img_idx * max_corners + ids[j];
                cam_landmarks[global_id] =
                    Eigen::Vector2d(corners[j].x, corners[j].y);
            }
        }
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 3: Compute relative pose for each spanning tree edge
// ─────────────────────────────────────────────────────────────────────────────

inline bool compute_relative_poses(
    const CalibrationTool::CalibConfig &config,
    const std::map<std::string, CameraIntrinsics> &intrinsics,
    const std::map<std::string, std::map<int, Eigen::Vector2d>> &landmarks,
    std::map<std::pair<int, int>, RelativePose> &relative_poses,
    std::string *status) {

    int num_cameras = (int)config.cam_ordered.size();

    // Build spanning tree edges: each second_view connects to its predecessor
    std::vector<std::pair<int, int>> edges;
    for (int i = 0; i < (int)config.second_view_order.size(); i++) {
        int cam_new = config.second_view_order[i];
        int cam_prev = (i == 0) ? config.first_view
                                : config.second_view_order[i - 1];
        edges.push_back({cam_prev, cam_new});
    }

    for (int e = 0; e < (int)edges.size(); e++) {
        auto [idx_a, idx_b] = edges[e];
        const std::string &serial_a = config.cam_ordered[idx_a];
        const std::string &serial_b = config.cam_ordered[idx_b];

        if (status)
            *status = "Computing relative poses (edge " +
                      std::to_string(e + 1) + "/" +
                      std::to_string(edges.size()) + "): " + serial_a +
                      " → " + serial_b + "...";

        const auto &lm_a = landmarks.at(serial_a);
        const auto &lm_b = landmarks.at(serial_b);
        const auto &intr_a = intrinsics.at(serial_a);
        const auto &intr_b = intrinsics.at(serial_b);

        // Find common landmark IDs
        std::vector<int> common_ids;
        for (const auto &[id, _] : lm_a) {
            if (lm_b.count(id))
                common_ids.push_back(id);
        }

        if (common_ids.size() < 20) {
            if (status)
                *status = "Error: only " + std::to_string(common_ids.size()) +
                          " common points between " + serial_a + " and " +
                          serial_b + " (need ≥20)";
            return false;
        }

        // Undistort 2D points
        std::vector<cv::Point2f> pts_a_cv, pts_b_cv;
        for (int id : common_ids) {
            Eigen::Vector2d ua =
                red_math::undistortPoint(lm_a.at(id), intr_a.K, intr_a.dist);
            Eigen::Vector2d ub =
                red_math::undistortPoint(lm_b.at(id), intr_b.K, intr_b.dist);
            pts_a_cv.push_back(cv::Point2f((float)ua.x(), (float)ua.y()));
            pts_b_cv.push_back(cv::Point2f((float)ub.x(), (float)ub.y()));
        }

        // Normalize points for fundamental matrix estimation by converting
        // undistorted pixel coords to normalized camera coords
        std::vector<cv::Point2f> pts_a_norm, pts_b_norm;
        for (int i = 0; i < (int)common_ids.size(); i++) {
            double fx_a = intr_a.K(0, 0), fy_a = intr_a.K(1, 1);
            double cx_a = intr_a.K(0, 2), cy_a = intr_a.K(1, 2);
            double fx_b = intr_b.K(0, 0), fy_b = intr_b.K(1, 1);
            double cx_b = intr_b.K(0, 2), cy_b = intr_b.K(1, 2);
            pts_a_norm.push_back(cv::Point2f(
                (float)((pts_a_cv[i].x - cx_a) / fx_a),
                (float)((pts_a_cv[i].y - cy_a) / fy_a)));
            pts_b_norm.push_back(cv::Point2f(
                (float)((pts_b_cv[i].x - cx_b) / fx_b),
                (float)((pts_b_cv[i].y - cy_b) / fy_b)));
        }

        // Find essential matrix directly from normalized points
        // (more robust than F → E conversion)
        cv::Mat inlier_mask;
        cv::Mat E = cv::findEssentialMat(pts_a_norm, pts_b_norm,
                                         cv::Mat::eye(3, 3, CV_64F),
                                         cv::RANSAC, 0.999, 0.001,
                                         inlier_mask);
        if (E.empty() || E.rows != 3 || E.cols != 3) {
            // Fallback: try with pixel coords and F matrix
            cv::Mat F = cv::findFundamentalMat(pts_a_cv, pts_b_cv,
                                               cv::FM_RANSAC, 3.0, 0.999,
                                               inlier_mask);
            if (F.empty() || F.rows != 3 || F.cols != 3) {
                if (status)
                    *status =
                        "Error: pose estimation failed for " + serial_a +
                        " → " + serial_b + " (" +
                        std::to_string(common_ids.size()) + " common pts)";
                return false;
            }
            cv::Mat Ka_cv, Kb_cv;
            cv::eigen2cv(intr_a.K, Ka_cv);
            cv::eigen2cv(intr_b.K, Kb_cv);
            E = Kb_cv.t() * F * Ka_cv;
        }

        // Decompose essential matrix → 4 candidate poses
        cv::Mat R1_cv, R2_cv, t_cv;
        cv::decomposeEssentialMat(E, R1_cv, R2_cv, t_cv);

        Eigen::Matrix3d R1, R2;
        Eigen::Vector3d t_dir;
        cv::cv2eigen(R1_cv, R1);
        cv::cv2eigen(R2_cv, R2);
        cv::cv2eigen(t_cv, t_dir);
        t_dir.normalize();

        // Four candidates: (R1, +t), (R1, -t), (R2, +t), (R2, -t)
        struct Candidate {
            Eigen::Matrix3d R;
            Eigen::Vector3d t;
        };
        Candidate candidates[4] = {
            {R1, t_dir}, {R1, -t_dir}, {R2, t_dir}, {R2, -t_dir}};

        int best_idx = 0;
        int best_count = -1;

        for (int c = 0; c < 4; c++) {
            // Projection matrices for triangulation
            auto P_a = red_math::projectionFromKRt(
                intr_a.K, Eigen::Matrix3d::Identity(),
                Eigen::Vector3d::Zero());
            auto P_b = red_math::projectionFromKRt(intr_b.K, candidates[c].R,
                                                   candidates[c].t);

            int positive_count = 0;
            for (int i = 0; i < (int)common_ids.size(); i++) {
                if (inlier_mask.at<uchar>(i) == 0)
                    continue;
                Eigen::Vector2d pa(pts_a_cv[i].x, pts_a_cv[i].y);
                Eigen::Vector2d pb(pts_b_cv[i].x, pts_b_cv[i].y);
                Eigen::Vector3d X = red_math::triangulatePoints(
                    {pa, pb}, {P_a, P_b});

                // Check positive depth in both cameras
                bool front_a = X.z() > 0;
                Eigen::Vector3d X_b = candidates[c].R * X + candidates[c].t;
                bool front_b = X_b.z() > 0;
                if (front_a && front_b)
                    positive_count++;
            }

            if (positive_count > best_count) {
                best_count = positive_count;
                best_idx = c;
            }
        }

        // Triangulate final points with the best pose
        RelativePose rp;
        rp.Rd = candidates[best_idx].R;
        rp.td = candidates[best_idx].t;

        auto P_a = red_math::projectionFromKRt(
            intr_a.K, Eigen::Matrix3d::Identity(), Eigen::Vector3d::Zero());
        auto P_b =
            red_math::projectionFromKRt(intr_b.K, rp.Rd, rp.td);

        for (int i = 0; i < (int)common_ids.size(); i++) {
            if (inlier_mask.at<uchar>(i) == 0)
                continue;
            Eigen::Vector2d pa(pts_a_cv[i].x, pts_a_cv[i].y);
            Eigen::Vector2d pb(pts_b_cv[i].x, pts_b_cv[i].y);
            Eigen::Vector3d X =
                red_math::triangulatePoints({pa, pb}, {P_a, P_b});

            // Skip points behind cameras
            Eigen::Vector3d X_b = rp.Rd * X + rp.td;
            if (X.z() <= 0 || X_b.z() <= 0)
                continue;

            rp.triang_points.push_back(X);
            rp.point_ids.push_back(common_ids[i]);
        }

        relative_poses[{idx_a, idx_b}] = std::move(rp);
    }

    if (status)
        *status = "Relative poses computed (" +
                  std::to_string(edges.size()) + " edges)";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 4: Chain pairwise poses along spanning tree
// ─────────────────────────────────────────────────────────────────────────────

inline bool concatenate_poses(
    const CalibrationTool::CalibConfig &config,
    const std::map<std::string, CameraIntrinsics> &intrinsics,
    const std::map<std::pair<int, int>, RelativePose> &relative_poses,
    std::vector<CameraPose> &poses, std::string *status) {

    int num_cameras = (int)config.cam_ordered.size();
    poses.resize(num_cameras);

    // Root camera: identity pose
    int root = config.first_view;
    const auto &root_intr = intrinsics.at(config.cam_ordered[root]);
    poses[root].R = Eigen::Matrix3d::Identity();
    poses[root].t = Eigen::Vector3d::Zero();
    poses[root].K = root_intr.K;
    poses[root].dist = root_intr.dist;

    // Accumulated world-frame 3D points for scale estimation
    std::map<int, Eigen::Vector3d> accumulated_pts;

    // Process edges in order
    for (int i = 0; i < (int)config.second_view_order.size(); i++) {
        int cam_new = config.second_view_order[i];
        int cam_prev =
            (i == 0) ? config.first_view : config.second_view_order[i - 1];

        const auto &rp = relative_poses.at({cam_prev, cam_new});
        const auto &intr_new = intrinsics.at(config.cam_ordered[cam_new]);

        double scale = 1.0;

        if (i > 0 && !accumulated_pts.empty()) {
            // Find common points between accumulated and locally-triangulated
            std::vector<std::pair<int, int>> common; // (accum_id, local_idx)
            for (int j = 0; j < (int)rp.point_ids.size(); j++) {
                if (accumulated_pts.count(rp.point_ids[j]))
                    common.push_back({rp.point_ids[j], j});
            }

            if (common.size() >= 2) {
                // Compute pairwise distance ratios
                const auto &R_prev = poses[cam_prev].R;
                const auto &t_prev = poses[cam_prev].t;
                std::vector<double> ratios;

                for (int j = 0; j < (int)common.size(); j++) {
                    for (int k = j + 1; k < (int)common.size(); k++) {
                        // Transform accumulated world points to cam_prev frame
                        Eigen::Vector3d Xj_cam =
                            R_prev * accumulated_pts[common[j].first] +
                            t_prev;
                        Eigen::Vector3d Xk_cam =
                            R_prev * accumulated_pts[common[k].first] +
                            t_prev;
                        double dist_world = (Xj_cam - Xk_cam).norm();

                        // Local triangulated distances
                        double dist_local =
                            (rp.triang_points[common[j].second] -
                             rp.triang_points[common[k].second])
                                .norm();

                        if (dist_local > 1e-10)
                            ratios.push_back(dist_world / dist_local);
                    }
                }

                if (!ratios.empty()) {
                    std::sort(ratios.begin(), ratios.end());
                    scale = ratios[ratios.size() / 2]; // median
                }
            }
        }

        // Chain pose
        const auto &R_prev = poses[cam_prev].R;
        const auto &t_prev = poses[cam_prev].t;
        poses[cam_new].R = rp.Rd * R_prev;
        poses[cam_new].t = rp.Rd * t_prev + scale * rp.td;
        poses[cam_new].K = intr_new.K;
        poses[cam_new].dist = intr_new.dist;

        // Transform locally-triangulated points to world frame and accumulate
        for (int j = 0; j < (int)rp.point_ids.size(); j++) {
            int pid = rp.point_ids[j];
            if (!accumulated_pts.count(pid)) {
                Eigen::Vector3d X_cam_true = scale * rp.triang_points[j];
                Eigen::Vector3d X_world =
                    R_prev.transpose() * (X_cam_true - t_prev);
                accumulated_pts[pid] = X_world;
            }
        }
    }

    if (status)
        *status = "Poses chained (" + std::to_string(num_cameras) +
                  " cameras, " + std::to_string(accumulated_pts.size()) +
                  " 3D points)";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 5: Bundle adjustment via Ceres
// ─────────────────────────────────────────────────────────────────────────────

// Ceres cost functor: reprojection error for one 2D observation.
struct ReprojectionCost {
    double obs_x, obs_y;

    ReprojectionCost(double x, double y) : obs_x(x), obs_y(y) {}

    template <typename T>
    bool operator()(const T *camera, const T *point, T *residuals) const {
        // camera[0..2]  = rvec (angle-axis)
        // camera[3..5]  = tvec
        // camera[6..9]  = fx, fy, cx, cy
        // camera[10..14]= k1, k2, p1, p2, k3

        // Rotate point
        T p[3];
        ceres::AngleAxisRotatePoint(camera, point, p);

        // Translate
        p[0] += camera[3];
        p[1] += camera[4];
        p[2] += camera[5];

        // Perspective division
        T xp = p[0] / p[2];
        T yp = p[1] / p[2];

        // Distortion
        T r2 = xp * xp + yp * yp;
        T r4 = r2 * r2;
        T r6 = r4 * r2;
        T k1 = camera[10], k2 = camera[11];
        T p1 = camera[12], p2 = camera[13];
        T k3 = camera[14];
        T radial = T(1) + k1 * r2 + k2 * r4 + k3 * r6;
        T xpp = xp * radial + T(2) * p1 * xp * yp +
                p2 * (r2 + T(2) * xp * xp);
        T ypp = yp * radial + p1 * (r2 + T(2) * yp * yp) +
                T(2) * p2 * xp * yp;

        // Pixel coordinates
        T fx = camera[6], fy = camera[7], cx = camera[8], cy = camera[9];
        T pred_x = fx * xpp + cx;
        T pred_y = fy * ypp + cy;

        residuals[0] = pred_x - T(obs_x);
        residuals[1] = pred_y - T(obs_y);
        return true;
    }

    static ceres::CostFunction *Create(double obs_x, double obs_y) {
        return new ceres::AutoDiffCostFunction<ReprojectionCost, 2, 15, 3>(
            new ReprojectionCost(obs_x, obs_y));
    }
};

inline bool bundle_adjust(
    const CalibrationTool::CalibConfig &config,
    const std::map<std::string, std::map<int, Eigen::Vector2d>> &landmarks,
    std::vector<CameraPose> &poses,
    std::map<int, Eigen::Vector3d> &points_3d, std::string *status) {

    int num_cameras = (int)config.cam_ordered.size();

    // Parse BA config
    double th_outliers_early = 1000.0;
    double th_outliers = 15.0;
    int max_nfev = 40;
    int max_nfev2 = 40;
    bool optimize_points = true;
    bool optimize_camera_params = true;
    bool use_bounds = false;
    std::vector<double> bounds_cp, bounds_pt;
    std::string loss_type = "linear";
    double f_scale = 1.0;

    if (!config.ba_config.is_null()) {
        const auto &ba = config.ba_config;
        th_outliers_early = ba.value("th_outliers_early", 1000.0);
        th_outliers = ba.value("th_outliers", 15.0);
        max_nfev = ba.value("max_nfev", 40);
        max_nfev2 = ba.value("max_nfev2", 40);
        optimize_points = ba.value("optimize_points", true);
        optimize_camera_params = ba.value("optimize_camera_params", true);
        use_bounds = ba.value("bounds", false);
        loss_type = ba.value("loss", std::string("linear"));
        f_scale = ba.value("f_scale", 1.0);
        if (ba.contains("bounds_cp") && ba["bounds_cp"].is_array())
            bounds_cp = ba["bounds_cp"].get<std::vector<double>>();
        if (ba.contains("bounds_pt") && ba["bounds_pt"].is_array())
            bounds_pt = ba["bounds_pt"].get<std::vector<double>>();
    }

    // Build initial 3D points from concatenated poses if not already populated
    // (points_3d should already be populated from concatenate_poses via the
    // accumulated points, but let's ensure we have a reasonable set)

    // Pack camera parameters: 15 doubles per camera
    // [rvec(3), tvec(3), fx, fy, cx, cy, k1, k2, p1, p2, k3]
    std::vector<std::array<double, 15>> camera_params(num_cameras);
    for (int i = 0; i < num_cameras; i++) {
        Eigen::Vector3d rvec = red_math::rotationMatrixToVector(poses[i].R);
        camera_params[i][0] = rvec.x();
        camera_params[i][1] = rvec.y();
        camera_params[i][2] = rvec.z();
        camera_params[i][3] = poses[i].t.x();
        camera_params[i][4] = poses[i].t.y();
        camera_params[i][5] = poses[i].t.z();
        camera_params[i][6] = poses[i].K(0, 0);  // fx
        camera_params[i][7] = poses[i].K(1, 1);  // fy
        camera_params[i][8] = poses[i].K(0, 2);  // cx
        camera_params[i][9] = poses[i].K(1, 2);  // cy
        camera_params[i][10] = poses[i].dist(0);  // k1
        camera_params[i][11] = poses[i].dist(1);  // k2
        camera_params[i][12] = poses[i].dist(2);  // p1
        camera_params[i][13] = poses[i].dist(3);  // p2
        camera_params[i][14] = poses[i].dist(4);  // k3
    }

    // Store initial values for bounds
    auto camera_params_init = camera_params;

    // Pack 3D point parameters: ordered by global ID
    std::vector<int> point_id_order;
    for (const auto &[id, _] : points_3d)
        point_id_order.push_back(id);
    std::sort(point_id_order.begin(), point_id_order.end());

    std::map<int, int> point_id_to_param_idx;
    std::vector<std::array<double, 3>> point_params(point_id_order.size());
    for (int i = 0; i < (int)point_id_order.size(); i++) {
        int pid = point_id_order[i];
        point_id_to_param_idx[pid] = i;
        const auto &pt = points_3d[pid];
        point_params[i] = {pt.x(), pt.y(), pt.z()};
    }

    auto point_params_init = point_params;

    // Build observations: (camera_idx, point_id, pixel_x, pixel_y)
    struct Observation {
        int cam_idx;
        int point_param_idx;
        double px, py;
    };
    std::vector<Observation> observations;

    for (int c = 0; c < num_cameras; c++) {
        const auto &serial = config.cam_ordered[c];
        auto it = landmarks.find(serial);
        if (it == landmarks.end())
            continue;
        for (const auto &[pid, pixel] : it->second) {
            if (point_id_to_param_idx.count(pid)) {
                observations.push_back(
                    {c, point_id_to_param_idx[pid], pixel.x(), pixel.y()});
            }
        }
    }

    if (observations.empty()) {
        if (status)
            *status = "Error: no observations for bundle adjustment";
        return false;
    }

    // Two-pass BA: first with loose outlier threshold, then tight
    for (int pass = 0; pass < 2; pass++) {
        double outlier_th = (pass == 0) ? th_outliers_early : th_outliers;
        int max_iter = (pass == 0) ? max_nfev : max_nfev2;

        if (status)
            *status = "Bundle adjustment (pass " + std::to_string(pass + 1) +
                      "/2, " + std::to_string(observations.size()) +
                      " observations)...";

        ceres::Problem problem;

        // Track which point indices have observations in this pass
        std::set<int> active_point_indices;
        for (const auto &obs : observations) {
            ceres::CostFunction *cost =
                ReprojectionCost::Create(obs.px, obs.py);

            ceres::LossFunction *loss = nullptr;
            if (loss_type == "huber")
                loss = new ceres::HuberLoss(f_scale);

            problem.AddResidualBlock(cost, loss,
                                     camera_params[obs.cam_idx].data(),
                                     point_params[obs.point_param_idx].data());
            active_point_indices.insert(obs.point_param_idx);
        }

        // Set bounds if configured.
        // bounds_cp[p] > 0  → constrain parameter to initial ± bound
        // bounds_cp[p] == 0 → fix parameter at initial value
        if (use_bounds && bounds_cp.size() >= 15) {
            for (int c = 0; c < num_cameras; c++) {
                // Collect indices to fix (bound == 0)
                std::vector<int> constant_indices;
                for (int p = 0; p < 15; p++) {
                    double b = bounds_cp[p];
                    if (b > 0) {
                        double init_val = camera_params_init[c][p];
                        problem.SetParameterLowerBound(
                            camera_params[c].data(), p, init_val - b);
                        problem.SetParameterUpperBound(
                            camera_params[c].data(), p, init_val + b);
                    } else if (b == 0) {
                        constant_indices.push_back(p);
                    }
                }
                // Fix zero-bounded parameters via SubsetManifold
                if (!constant_indices.empty()) {
                    problem.SetManifold(
                        camera_params[c].data(),
                        new ceres::SubsetManifold(15, constant_indices));
                }
            }
        }

        if (use_bounds && bounds_pt.size() >= 3 && optimize_points) {
            for (int i : active_point_indices) {
                for (int p = 0; p < 3; p++) {
                    double b = bounds_pt[p];
                    if (b > 0) {
                        double init_val = point_params_init[i][p];
                        problem.SetParameterLowerBound(
                            point_params[i].data(), p, init_val - b);
                        problem.SetParameterUpperBound(
                            point_params[i].data(), p, init_val + b);
                    }
                }
            }
        }

        // Fix parameters if not optimizing
        if (!optimize_camera_params) {
            for (int c = 0; c < num_cameras; c++)
                problem.SetParameterBlockConstant(camera_params[c].data());
        }
        if (!optimize_points) {
            for (int i = 0; i < (int)point_params.size(); i++)
                problem.SetParameterBlockConstant(point_params[i].data());
        }

        // Solver options
        ceres::Solver::Options options;
        options.linear_solver_type = ceres::SPARSE_SCHUR;
        options.max_num_iterations = max_iter;
        options.function_tolerance = 1e-8;
        options.parameter_tolerance = 1e-8;
        options.minimizer_progress_to_stdout = false;
        options.num_threads = std::max(1, (int)std::thread::hardware_concurrency());

        ceres::Solver::Summary summary;
        ceres::Solve(options, &problem, &summary);

        printf("BA pass %d: %s  initial_cost=%.2f  final_cost=%.2f  "
               "iterations=%d  time=%.2fs\n",
               pass + 1, summary.IsSolutionUsable() ? "CONVERGED" : "FAILED",
               summary.initial_cost, summary.final_cost,
               (int)summary.iterations.size(), summary.total_time_in_seconds);

        // Reject outliers after first pass
        if (pass == 0) {
            std::vector<Observation> inlier_obs;
            for (const auto &obs : observations) {
                // Compute reprojection error
                Eigen::Vector3d rvec(camera_params[obs.cam_idx][0],
                                     camera_params[obs.cam_idx][1],
                                     camera_params[obs.cam_idx][2]);
                Eigen::Vector3d tvec(camera_params[obs.cam_idx][3],
                                     camera_params[obs.cam_idx][4],
                                     camera_params[obs.cam_idx][5]);
                Eigen::Matrix3d K_cam = Eigen::Matrix3d::Identity();
                K_cam(0, 0) = camera_params[obs.cam_idx][6];
                K_cam(1, 1) = camera_params[obs.cam_idx][7];
                K_cam(0, 2) = camera_params[obs.cam_idx][8];
                K_cam(1, 2) = camera_params[obs.cam_idx][9];
                Eigen::Matrix<double, 5, 1> d;
                d << camera_params[obs.cam_idx][10],
                    camera_params[obs.cam_idx][11],
                    camera_params[obs.cam_idx][12],
                    camera_params[obs.cam_idx][13],
                    camera_params[obs.cam_idx][14];

                auto &pp = point_params[obs.point_param_idx];
                Eigen::Vector3d pt3d(pp[0], pp[1], pp[2]);

                Eigen::Vector2d projected =
                    red_math::projectPoint(pt3d, rvec, tvec, K_cam, d);
                double err = std::sqrt(std::pow(projected.x() - obs.px, 2) +
                                       std::pow(projected.y() - obs.py, 2));

                if (err < outlier_th)
                    inlier_obs.push_back(obs);
            }

            int removed = (int)observations.size() - (int)inlier_obs.size();
            observations = std::move(inlier_obs);

            if (status)
                *status = "BA pass 1 done. Removed " +
                          std::to_string(removed) + " outliers (threshold=" +
                          std::to_string(outlier_th).substr(0, 6) + ")";
        }
    }

    // Log BA intrinsic changes for diagnostics
    printf("\n=== Bundle Adjustment Intrinsic Changes ===\n");
    printf("%-12s  %8s %8s %8s %8s  |  %8s %8s %8s %8s\n",
           "Camera", "fx_init", "fy_init", "cx_init", "cy_init",
           "dfx", "dfy", "dcx", "dcy");
    for (int i = 0; i < num_cameras; i++) {
        printf("%-12s  %8.1f %8.1f %8.1f %8.1f  |  %+8.2f %+8.2f %+8.2f %+8.2f\n",
               config.cam_ordered[i].c_str(),
               camera_params_init[i][6], camera_params_init[i][7],
               camera_params_init[i][8], camera_params_init[i][9],
               camera_params[i][6] - camera_params_init[i][6],
               camera_params[i][7] - camera_params_init[i][7],
               camera_params[i][8] - camera_params_init[i][8],
               camera_params[i][9] - camera_params_init[i][9]);
    }
    printf("============================================\n\n");

    // Unpack results back into poses and points_3d
    for (int i = 0; i < num_cameras; i++) {
        Eigen::Vector3d rvec(camera_params[i][0], camera_params[i][1],
                             camera_params[i][2]);
        poses[i].R = red_math::rotationVectorToMatrix(rvec);
        poses[i].t =
            Eigen::Vector3d(camera_params[i][3], camera_params[i][4],
                            camera_params[i][5]);
        poses[i].K = Eigen::Matrix3d::Identity();
        poses[i].K(0, 0) = camera_params[i][6];
        poses[i].K(1, 1) = camera_params[i][7];
        poses[i].K(0, 2) = camera_params[i][8];
        poses[i].K(1, 2) = camera_params[i][9];
        poses[i].dist << camera_params[i][10], camera_params[i][11],
            camera_params[i][12], camera_params[i][13], camera_params[i][14];
    }

    for (int i = 0; i < (int)point_id_order.size(); i++) {
        points_3d[point_id_order[i]] =
            Eigen::Vector3d(point_params[i][0], point_params[i][1],
                            point_params[i][2]);
    }

    // Compute mean reprojection error
    double total_err = 0.0;
    int total_obs = 0;
    for (const auto &obs : observations) {
        Eigen::Vector3d rvec(camera_params[obs.cam_idx][0],
                             camera_params[obs.cam_idx][1],
                             camera_params[obs.cam_idx][2]);
        Eigen::Vector3d tvec(camera_params[obs.cam_idx][3],
                             camera_params[obs.cam_idx][4],
                             camera_params[obs.cam_idx][5]);
        auto &pp = point_params[obs.point_param_idx];
        Eigen::Vector3d pt3d(pp[0], pp[1], pp[2]);

        Eigen::Vector2d projected =
            red_math::projectPoint(pt3d, rvec, tvec, poses[obs.cam_idx].K,
                                   poses[obs.cam_idx].dist);
        double err = std::sqrt(std::pow(projected.x() - obs.px, 2) +
                               std::pow(projected.y() - obs.py, 2));
        total_err += err;
        total_obs++;
    }

    double mean_err = (total_obs > 0) ? (total_err / total_obs) : 0.0;
    if (status)
        *status = "Bundle adjustment complete. Mean reproj error: " +
                  std::to_string(mean_err).substr(0, 5) + " px (" +
                  std::to_string(total_obs) + " observations)";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 6: Global registration (Procrustes alignment to world frame)
// ─────────────────────────────────────────────────────────────────────────────

inline bool global_registration(
    const CalibrationTool::CalibConfig &config,
    std::vector<CameraPose> &poses,
    std::map<int, Eigen::Vector3d> &points_3d, std::string *status) {

    if (config.gt_pts.empty() || config.world_coordinate_imgs.empty()) {
        if (status)
            *status = "Skipping global registration (no gt_pts or "
                      "world_coordinate_imgs in config)";
        return true; // Not an error, just skip
    }

    int max_corners =
        (config.charuco_setup.w - 1) * (config.charuco_setup.h - 1);

    // Get sorted image numbers to map image name → sorted index
    auto image_numbers =
        get_sorted_image_numbers(config.img_path, config.cam_ordered[0]);
    std::map<int, int> img_num_to_idx;
    for (int i = 0; i < (int)image_numbers.size(); i++)
        img_num_to_idx[image_numbers[i]] = i;

    // Collect corresponding pairs: BA 3D point ↔ world coordinate
    std::vector<Eigen::Vector3d> src_pts; // BA frame
    std::vector<Eigen::Vector3d> dst_pts; // world frame (gt_pts)

    for (const auto &img_name : config.world_coordinate_imgs) {
        int img_num = std::stoi(img_name);
        auto it = img_num_to_idx.find(img_num);
        if (it == img_num_to_idx.end())
            continue;
        int img_idx = it->second;

        auto gt_it = config.gt_pts.find(img_name);
        if (gt_it == config.gt_pts.end())
            continue;
        const auto &gt = gt_it->second;

        for (int corner_id = 0;
             corner_id < max_corners && corner_id < (int)gt.size();
             corner_id++) {
            int global_id = img_idx * max_corners + corner_id;
            auto pt_it = points_3d.find(global_id);
            if (pt_it != points_3d.end()) {
                src_pts.push_back(pt_it->second);
                dst_pts.push_back(Eigen::Vector3d(gt[corner_id][0],
                                                  gt[corner_id][1],
                                                  gt[corner_id][2]));
            }
        }
    }

    if (src_pts.size() < 3) {
        if (status)
            *status = "Error: only " + std::to_string(src_pts.size()) +
                      " matching points for global registration (need ≥3)";
        return false;
    }

    if (status)
        *status = "Aligning to world frame (" +
                  std::to_string(src_pts.size()) + " ground truth points)...";

    // Procrustes alignment: find R, s, t such that dst ≈ s*R*src + t
    int n = (int)src_pts.size();

    // 1. Compute centroids
    Eigen::Vector3d src_mean = Eigen::Vector3d::Zero();
    Eigen::Vector3d dst_mean = Eigen::Vector3d::Zero();
    for (int i = 0; i < n; i++) {
        src_mean += src_pts[i];
        dst_mean += dst_pts[i];
    }
    src_mean /= n;
    dst_mean /= n;

    // 2. Center the points
    Eigen::MatrixXd S(n, 3), D(n, 3);
    for (int i = 0; i < n; i++) {
        S.row(i) = (src_pts[i] - src_mean).transpose();
        D.row(i) = (dst_pts[i] - dst_mean).transpose();
    }

    // 3. Cross-covariance matrix M = D^T * S
    Eigen::Matrix3d M = D.transpose() * S;

    // 4. SVD
    Eigen::JacobiSVD<Eigen::Matrix3d> svd(
        M, Eigen::ComputeFullU | Eigen::ComputeFullV);
    Eigen::Matrix3d U = svd.matrixU();
    Eigen::Matrix3d V = svd.matrixV();

    // Ensure proper rotation (det > 0)
    Eigen::Matrix3d R_reg = U * V.transpose();
    if (R_reg.determinant() < 0) {
        V.col(2) *= -1;
        R_reg = U * V.transpose();
    }

    // 5. Scale: s = trace(Sigma) / trace(S^T * S)
    double src_var = 0.0;
    for (int i = 0; i < n; i++)
        src_var += S.row(i).squaredNorm();
    double scale_reg = svd.singularValues().sum() / src_var;

    // 6. Translation: t = dst_mean - s * R * src_mean
    Eigen::Vector3d t_reg = dst_mean - scale_reg * R_reg * src_mean;

    // Apply transformation to all camera poses and 3D points
    // Original: X_cam = R_old * X_world_old + t_old
    // New world: X_world_new = scale * R_reg * X_world_old + t_reg
    // So: X_world_old = R_reg^T * (X_world_new - t_reg) / scale
    // X_cam = R_old * R_reg^T * (X_world_new - t_reg) / scale + t_old
    //       = (R_old * R_reg^T / scale) * X_world_new
    //         + t_old - R_old * R_reg^T * t_reg / scale
    // But that mixes scale into the extrinsics. The standard approach:
    //
    // We want the new extrinsics (R_new, t_new) such that:
    //   X_cam = R_new * X_world_new + t_new
    // where X_world_new = s * R_reg * X_world_old + t_reg
    //
    // X_cam = R_old * X_world_old + t_old
    //       = R_old * R_reg^T * (X_world_new - t_reg) / s + t_old
    //       = (R_old * R_reg^T / s) * X_world_new
    //         + t_old - R_old * R_reg^T * t_reg / s
    //
    // So: R_new = R_old * R_reg^T  (ignoring scale in R)
    //     t_new = scale_reg * R_new * ( ... )
    //
    // Actually the cleaner formulation: the camera center in old world coords
    // is C_old = -R_old^T * t_old. In new world coords:
    // C_new = scale_reg * R_reg * C_old + t_reg
    // Then: t_new = -R_new * C_new
    //       R_new = R_old * R_reg^T

    for (int i = 0; i < (int)poses.size(); i++) {
        // Camera center in old world frame
        Eigen::Vector3d C_old = -poses[i].R.transpose() * poses[i].t;
        // Transform to new world frame
        Eigen::Vector3d C_new = scale_reg * R_reg * C_old + t_reg;
        // New rotation
        Eigen::Matrix3d R_new = poses[i].R * R_reg.transpose();
        // New translation
        Eigen::Vector3d t_new = -R_new * C_new;

        poses[i].R = R_new;
        poses[i].t = t_new;
    }

    // Transform 3D points
    for (auto &[id, pt] : points_3d) {
        pt = scale_reg * R_reg * pt + t_reg;
    }

    // Report registration error
    double reg_err = 0.0;
    for (int i = 0; i < n; i++) {
        Eigen::Vector3d transformed =
            scale_reg * R_reg * src_pts[i] + t_reg;
        reg_err += (transformed - dst_pts[i]).norm();
    }
    reg_err /= n;

    if (status)
        *status = "Global registration complete. Mean error: " +
                  std::to_string(reg_err).substr(0, 6) + " mm (" +
                  std::to_string(n) + " points, scale=" +
                  std::to_string(scale_reg).substr(0, 6) + ")";
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Step 7: Write per-camera YAML calibration files
// ─────────────────────────────────────────────────────────────────────────────

inline bool write_calibration(
    const std::vector<CameraPose> &poses,
    const std::vector<std::string> &cam_names,
    const std::string &output_folder, int image_width, int image_height,
    std::string *status) {

    namespace fs = std::filesystem;

    // Create output directory
    std::error_code ec;
    fs::create_directories(output_folder, ec);
    if (ec) {
        if (status)
            *status = "Error: cannot create output folder: " + ec.message();
        return false;
    }

    for (int i = 0; i < (int)poses.size(); i++) {
        std::string filename =
            output_folder + "/Cam" + cam_names[i] + ".yaml";

        try {
            opencv_yaml::YamlWriter writer(filename);

            writer.writeScalar("image_width", image_width);
            writer.writeScalar("image_height", image_height);

            // Camera matrix (3×3)
            writer.writeMatrix("camera_matrix", poses[i].K);

            // Distortion coefficients (5×1)
            Eigen::MatrixXd dist_mat(5, 1);
            for (int j = 0; j < 5; j++)
                dist_mat(j, 0) = poses[i].dist(j);
            writer.writeMatrix("distortion_coefficients", dist_mat);

            // Translation (3×1)
            Eigen::MatrixXd t_mat(3, 1);
            t_mat(0, 0) = poses[i].t.x();
            t_mat(1, 0) = poses[i].t.y();
            t_mat(2, 0) = poses[i].t.z();
            writer.writeMatrix("tc_ext", t_mat);

            // Rotation (3×3)
            writer.writeMatrix("rc_ext", poses[i].R);

            writer.close();
        } catch (const std::exception &e) {
            if (status)
                *status =
                    "Error writing " + filename + ": " + std::string(e.what());
            return false;
        }
    }

    if (status)
        *status = "Wrote " + std::to_string(poses.size()) +
                  " calibration files to " + output_folder;
    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Write intermediate output (matches multiview_calib output/ structure)
// ─────────────────────────────────────────────────────────────────────────────

inline bool write_intermediate_output(
    const CalibrationTool::CalibConfig &config,
    const std::map<std::string, CameraIntrinsics> &intrinsics,
    const std::map<std::string, std::map<int, Eigen::Vector2d>> &landmarks,
    const std::vector<CameraPose> &poses,
    const std::map<int, Eigen::Vector3d> &points_3d,
    const std::string &output_folder) {

    namespace fs = std::filesystem;
    std::string output_dir = output_folder + "/summary_data";
    std::string ba_dir = output_dir + "/bundle_adjustment";
    std::string intr_dir = output_dir + "/intrinsics";

    std::error_code ec;
    fs::create_directories(ba_dir, ec);
    fs::create_directories(intr_dir, ec);

    int num_cameras = (int)config.cam_ordered.size();

    // ── intrinsics.json ──
    {
        nlohmann::json j;
        for (const auto &serial : config.cam_ordered) {
            auto it = intrinsics.find(serial);
            if (it == intrinsics.end()) continue;
            const auto &intr = it->second;
            nlohmann::json cam_j;
            cam_j["K"] = {
                {intr.K(0,0), intr.K(0,1), intr.K(0,2)},
                {intr.K(1,0), intr.K(1,1), intr.K(1,2)},
                {intr.K(2,0), intr.K(2,1), intr.K(2,2)}
            };
            cam_j["dist"] = {intr.dist(0), intr.dist(1), intr.dist(2),
                             intr.dist(3), intr.dist(4)};
            cam_j["reprojection_error"] = intr.reproj_error;
            cam_j["image_width"] = intr.image_width;
            cam_j["image_height"] = intr.image_height;
            j[serial] = cam_j;
        }
        std::ofstream f(output_dir + "/intrinsics.json");
        f << j.dump(2);
    }

    // ── Per-camera intrinsic YAML files ──
    for (const auto &serial : config.cam_ordered) {
        auto it = intrinsics.find(serial);
        if (it == intrinsics.end()) continue;
        const auto &intr = it->second;
        std::string filename = intr_dir + "/" + serial + ".yaml";
        opencv_yaml::YamlWriter writer(filename);
        writer.writeScalar("image_width", intr.image_width);
        writer.writeScalar("image_height", intr.image_height);
        writer.writeMatrix("camera_matrix", intr.K);
        Eigen::MatrixXd dist_mat(5, 1);
        for (int j = 0; j < 5; j++) dist_mat(j, 0) = intr.dist(j);
        writer.writeMatrix("distortion_coefficients", dist_mat);
        writer.close();
    }

    // ── landmarks.json ──
    {
        nlohmann::json j;
        for (const auto &serial : config.cam_ordered) {
            auto it = landmarks.find(serial);
            if (it == landmarks.end()) continue;
            nlohmann::json cam_j;
            std::vector<int> ids;
            std::vector<std::vector<double>> pts;
            for (const auto &[pid, pixel] : it->second) {
                ids.push_back(pid);
                pts.push_back({pixel.x(), pixel.y()});
            }
            cam_j["ids"] = ids;
            cam_j["landmarks"] = pts;
            j[serial] = cam_j;
        }
        std::ofstream f(output_dir + "/landmarks.json");
        f << j.dump(2);
    }

    // ── bundle_adjustment/ba_poses.json ──
    {
        nlohmann::json j;
        for (int i = 0; i < num_cameras; i++) {
            const auto &serial = config.cam_ordered[i];
            nlohmann::json cam_j;
            // R as 3x3 row-major array
            nlohmann::json R_j = nlohmann::json::array();
            for (int r = 0; r < 3; r++) {
                nlohmann::json row = nlohmann::json::array();
                for (int c = 0; c < 3; c++) row.push_back(poses[i].R(r, c));
                R_j.push_back(row);
            }
            cam_j["R"] = R_j;
            cam_j["t"] = {poses[i].t.x(), poses[i].t.y(), poses[i].t.z()};
            nlohmann::json K_j = nlohmann::json::array();
            for (int r = 0; r < 3; r++) {
                nlohmann::json row = nlohmann::json::array();
                for (int c = 0; c < 3; c++) row.push_back(poses[i].K(r, c));
                K_j.push_back(row);
            }
            cam_j["K"] = K_j;
            cam_j["dist"] = {poses[i].dist(0), poses[i].dist(1),
                             poses[i].dist(2), poses[i].dist(3),
                             poses[i].dist(4)};
            j[serial] = cam_j;
        }
        std::ofstream f(ba_dir + "/ba_poses.json");
        f << j.dump(2);
    }

    // ── bundle_adjustment/ba_points.json ──
    {
        nlohmann::json j;
        for (const auto &[id, pt] : points_3d) {
            j[std::to_string(id)] = {pt.x(), pt.y(), pt.z()};
        }
        std::ofstream f(ba_dir + "/ba_points.json");
        f << j.dump(2);
    }

    // ── bundle_adjustment/bundle_adjustment.log ──
    {
        std::ofstream f(ba_dir + "/bundle_adjustment.log");
        f << "Reprojection errors (mean+-std pixels):\n";

        double global_total = 0;
        int global_count = 0;

        for (int c = 0; c < num_cameras; c++) {
            const auto &serial = config.cam_ordered[c];
            auto lm_it = landmarks.find(serial);
            if (lm_it == landmarks.end()) continue;

            Eigen::Vector3d rvec =
                red_math::rotationMatrixToVector(poses[c].R);

            std::vector<double> errs;
            for (const auto &[pid, pixel] : lm_it->second) {
                auto pt_it = points_3d.find(pid);
                if (pt_it == points_3d.end()) continue;
                Eigen::Vector2d projected = red_math::projectPoint(
                    pt_it->second, rvec, poses[c].t, poses[c].K,
                    poses[c].dist);
                errs.push_back((projected - pixel).norm());
            }

            if (errs.empty()) continue;

            double mean = 0;
            for (double e : errs) mean += e;
            mean /= errs.size();

            double var = 0;
            for (double e : errs) var += (e - mean) * (e - mean);
            double std_dev = std::sqrt(var / errs.size());

            // Median
            std::sort(errs.begin(), errs.end());
            double median = errs[errs.size() / 2];

            f << "\t " << serial << " n_points=" << errs.size()
              << ": " << std::fixed << std::setprecision(3)
              << mean << "+-" << std_dev
              << " (median=" << median << ")\n";

            global_total += mean * errs.size();
            global_count += errs.size();
        }

        double global_mean = (global_count > 0) ?
            (global_total / global_count) : 0.0;
        f << "Average absolute residual: " << std::fixed
          << std::setprecision(2) << global_mean
          << " over " << global_count << " points.\n";
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Top-level orchestrator: runs Steps 1-7 sequentially
// ─────────────────────────────────────────────────────────────────────────────

inline CalibrationResult
run_full_pipeline(const CalibrationTool::CalibConfig &config,
                  const std::string &base_folder,
                  std::string *status) {
    CalibrationResult result;
    result.cam_names = config.cam_ordered;

    // Create timestamped output folder inside base_folder
    auto now = std::chrono::system_clock::now();
    std::time_t t = std::chrono::system_clock::to_time_t(now);
    std::tm tstruct;
    localtime_r(&t, &tstruct);
    char tbuf[64];
    std::strftime(tbuf, sizeof(tbuf), "%Y_%m_%d_%H_%M_%S", &tstruct);
    std::string output_folder = base_folder + "/" + tbuf;

    {
        namespace fs = std::filesystem;
        std::error_code ec;
        fs::create_directories(output_folder, ec);
        if (ec) {
            result.error = "Cannot create output folder: " + ec.message();
            return result;
        }
    }

    // Step 1: Detect ChArUco corners + calibrate intrinsics
    if (status)
        *status = "Step 1/7: Detecting ChArUco corners...";
    std::map<std::string, CameraIntrinsics> intrinsics;
    if (!detect_and_calibrate_intrinsics(config, intrinsics, status)) {
        result.error = status ? *status : "Intrinsic calibration failed";
        return result;
    }

    // Get image dimensions from first camera
    result.image_width = intrinsics.begin()->second.image_width;
    result.image_height = intrinsics.begin()->second.image_height;

    // Step 2: Build landmarks
    if (status)
        *status = "Step 2/7: Building landmarks...";
    std::map<std::string, std::map<int, Eigen::Vector2d>> landmarks;
    build_landmarks(config, intrinsics, landmarks);

    // Step 3: Compute relative poses
    if (status)
        *status = "Step 3/7: Computing relative poses...";
    std::map<std::pair<int, int>, RelativePose> relative_poses;
    if (!compute_relative_poses(config, intrinsics, landmarks, relative_poses,
                                status)) {
        result.error = status ? *status : "Relative pose estimation failed";
        return result;
    }

    // Step 4: Chain poses
    if (status)
        *status = "Step 4/7: Chaining poses along spanning tree...";
    std::vector<CameraPose> poses;
    if (!concatenate_poses(config, intrinsics, relative_poses, poses,
                           status)) {
        result.error = status ? *status : "Pose concatenation failed";
        return result;
    }

    // Collect all accumulated 3D points for BA
    // Re-triangulate using the chained poses for a complete point set
    std::map<int, Eigen::Vector3d> points_3d;
    {
        int max_corners =
            (config.charuco_setup.w - 1) * (config.charuco_setup.h - 1);

        // For each 3D point, triangulate from all cameras that see it
        std::map<int, std::vector<std::pair<int, Eigen::Vector2d>>>
            point_observations;
        for (int c = 0; c < (int)config.cam_ordered.size(); c++) {
            const auto &serial = config.cam_ordered[c];
            auto it = landmarks.find(serial);
            if (it == landmarks.end())
                continue;
            for (const auto &[pid, pixel] : it->second) {
                point_observations[pid].push_back({c, pixel});
            }
        }

        for (const auto &[pid, obs] : point_observations) {
            if (obs.size() < 2)
                continue;
            std::vector<Eigen::Vector2d> pts2d;
            std::vector<Eigen::Matrix<double, 3, 4>> Ps;
            for (const auto &[cam_idx, pixel] : obs) {
                // Undistort the point
                Eigen::Vector2d und = red_math::undistortPoint(
                    pixel, poses[cam_idx].K, poses[cam_idx].dist);
                pts2d.push_back(und);
                Ps.push_back(red_math::projectionFromKRt(
                    poses[cam_idx].K, poses[cam_idx].R, poses[cam_idx].t));
            }
            Eigen::Vector3d X = red_math::triangulatePoints(pts2d, Ps);
            points_3d[pid] = X;
        }
    }

    // Step 5: Bundle adjustment
    if (status)
        *status = "Step 5/7: Bundle adjustment...";
    if (!bundle_adjust(config, landmarks, poses, points_3d, status)) {
        result.error = status ? *status : "Bundle adjustment failed";
        return result;
    }

    // Step 6: Global registration
    if (status)
        *status = "Step 6/7: Global registration...";
    if (!global_registration(config, poses, points_3d, status)) {
        result.error = status ? *status : "Global registration failed";
        return result;
    }

    // Step 7: Write calibration files
    if (status)
        *status = "Step 7/7: Writing calibration files...";
    if (!write_calibration(poses, config.cam_ordered, output_folder,
                           result.image_width, result.image_height, status)) {
        result.error = status ? *status : "Failed to write calibration files";
        return result;
    }

    // Write summary data for validation/comparison
    write_intermediate_output(config, intrinsics, landmarks, poses,
                              points_3d, output_folder);

    // Compute final mean reprojection error
    double total_err = 0.0;
    int total_obs = 0;
    for (int c = 0; c < (int)config.cam_ordered.size(); c++) {
        const auto &serial = config.cam_ordered[c];
        auto lm_it = landmarks.find(serial);
        if (lm_it == landmarks.end())
            continue;
        Eigen::Vector3d rvec = red_math::rotationMatrixToVector(poses[c].R);
        for (const auto &[pid, pixel] : lm_it->second) {
            auto pt_it = points_3d.find(pid);
            if (pt_it == points_3d.end())
                continue;
            Eigen::Vector2d projected = red_math::projectPoint(
                pt_it->second, rvec, poses[c].t, poses[c].K, poses[c].dist);
            double err = (projected - pixel).norm();
            total_err += err;
            total_obs++;
        }
    }
    result.mean_reproj_error =
        (total_obs > 0) ? (total_err / total_obs) : 0.0;

    result.output_folder = output_folder;
    result.cameras = std::move(poses);
    result.points_3d = std::move(points_3d);
    result.success = true;

    if (status)
        *status = "Calibration complete! Mean reproj error: " +
                  std::to_string(result.mean_reproj_error).substr(0, 5) +
                  " px (" + std::to_string(total_obs) + " observations, " +
                  std::to_string(result.cameras.size()) + " cameras)";

    return result;
}

} // namespace CalibrationPipeline
