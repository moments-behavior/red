// test_calib_debug.cpp — Standalone diagnostic for failing camera pair.
// Loads config, runs Steps 1-2 of the calibration pipeline, then exercises
// findEssentialMat / findFundamentalMat on the pair (2006054 -> 2002486)
// with verbose output to help debug failures.
//
// Build: cmake target "test_calib_debug" (no ImGui/Metal needed).
// Run:   ./test_calib_debug

#define STB_IMAGE_IMPLEMENTATION
#include "calibration_pipeline.h"

#include <iomanip>
#include <iostream>

static const char *CONFIG_PATH =
    "/Users/johnsonr/datasets/QuanShare/CalibrationDataset/2026_03_01/config.json";

// The failing pair: cam_ordered[1] -> cam_ordered[2]
static const char *SERIAL_A = "2006054";
static const char *SERIAL_B = "2002486";

static void print_matrix(const std::string &label, const cv::Mat &m) {
    std::cout << label << " (" << m.rows << "x" << m.cols
              << ", type=" << m.type() << "):\n";
    for (int r = 0; r < m.rows; r++) {
        std::cout << "  [";
        for (int c = 0; c < m.cols; c++) {
            if (c > 0) std::cout << ", ";
            std::cout << std::setw(14) << std::setprecision(8)
                      << m.at<double>(r, c);
        }
        std::cout << "]\n";
    }
}

static void print_eigen3(const std::string &label, const Eigen::Matrix3d &m) {
    std::cout << label << ":\n";
    for (int r = 0; r < 3; r++) {
        std::cout << "  [";
        for (int c = 0; c < 3; c++) {
            if (c > 0) std::cout << ", ";
            std::cout << std::setw(14) << std::setprecision(8) << m(r, c);
        }
        std::cout << "]\n";
    }
}

int main() {
    std::cout << "=== Calibration debug: " << SERIAL_A << " -> " << SERIAL_B
              << " ===\n\n";

    // ---- Load config ----
    CalibrationTool::CalibConfig config;
    std::string error;
    if (!CalibrationTool::parse_config(CONFIG_PATH, config, error)) {
        std::cerr << "Failed to parse config: " << error << "\n";
        return 1;
    }
    std::cout << "Config loaded. " << config.cam_ordered.size()
              << " cameras, img_path=" << config.img_path << "\n";
    std::cout << "cam_ordered:";
    for (const auto &s : config.cam_ordered) std::cout << " " << s;
    std::cout << "\n";
    std::cout << "first_view=" << config.first_view << ", second_view_order:";
    for (int v : config.second_view_order) std::cout << " " << v;
    std::cout << "\n";
    std::cout << "charuco: " << config.charuco_setup.w << "x"
              << config.charuco_setup.h
              << ", sq=" << config.charuco_setup.square_side_length
              << ", mk=" << config.charuco_setup.marker_side_length
              << ", dict=" << config.charuco_setup.dictionary << "\n\n";

    // ---- Direct single-camera calibration test ----
    {
        std::cout << "--- Direct calibration test for " << SERIAL_A << " ---\n";
        auto &cs = config.charuco_setup;
        auto dictionary = cv::aruco::getPredefinedDictionary(
            CalibrationPipeline::aruco_dict_from_id(cs.dictionary));
        auto board = cv::makePtr<cv::aruco::CharucoBoard>(
            cv::Size(cs.w, cs.h), cs.square_side_length,
            cs.marker_side_length, dictionary);
        cv::aruco::CharucoDetector detector(*board);

        auto img_nums = CalibrationPipeline::get_sorted_image_numbers(
            config.img_path, SERIAL_A);
        auto ext = CalibrationPipeline::get_image_extension(
            config.img_path, SERIAL_A);

        std::vector<std::vector<cv::Point3f>> all_obj;
        std::vector<std::vector<cv::Point2f>> all_img;
        cv::Size imsz;

        for (int n : img_nums) {
            std::string f = config.img_path + "/" + std::string(SERIAL_A) +
                            "_" + std::to_string(n) + ext;
            int w = 0, h = 0, ch = 0;
            unsigned char *px = stbi_load(f.c_str(), &w, &h, &ch, 1);
            if (!px) continue;
            cv::Mat im(h, w, CV_8UC1, px);
            if (imsz.width == 0) imsz = cv::Size(w, h);

            std::vector<cv::Point2f> corners;
            std::vector<int> ids;
            detector.detectBoard(im, corners, ids);
            if ((int)ids.size() < 6) { stbi_image_free(px); continue; }

            std::vector<cv::Point3f> op;
            std::vector<cv::Point2f> ip;
            board->matchImagePoints(corners, ids, op, ip);

            std::cout << "  img " << n << ": detectBoard=" << ids.size()
                      << " corners, matchImagePoints=" << op.size()
                      << " obj_pts, " << ip.size() << " img_pts\n";
            if ((int)op.size() >= 6) {
                all_obj.push_back(op);
                all_img.push_back(ip);
            }
            stbi_image_free(px);
            if (all_obj.size() >= 5) break; // just test a few
        }

        std::cout << "  Total views for calibration: " << all_obj.size() << "\n";
        std::cout << "  Image size: " << imsz.width << "x" << imsz.height << "\n";

        cv::Mat K = cv::Mat::eye(3, 3, CV_64F);
        K.at<double>(0, 0) = imsz.width;
        K.at<double>(1, 1) = imsz.width;
        K.at<double>(0, 2) = imsz.width / 2.0;
        K.at<double>(1, 2) = imsz.height / 2.0;
        cv::Mat dist;

        std::cout << "  K before: fx=" << K.at<double>(0,0) << " fy="
                  << K.at<double>(1,1) << " cx=" << K.at<double>(0,2)
                  << " cy=" << K.at<double>(1,2) << "\n";

        cv::Mat rvecs, tvecs;
        double err = cv::calibrateCamera(all_obj, all_img, imsz, K, dist,
                                         rvecs, tvecs,
                                         cv::CALIB_USE_INTRINSIC_GUESS |
                                         cv::CALIB_FIX_ASPECT_RATIO);

        std::cout << "  K after:  fx=" << K.at<double>(0,0) << " fy="
                  << K.at<double>(1,1) << " cx=" << K.at<double>(0,2)
                  << " cy=" << K.at<double>(1,2) << "\n";
        std::cout << "  dist: ";
        for (int i = 0; i < std::min((int)dist.total(), 5); i++)
            std::cout << dist.at<double>(i) << " ";
        std::cout << "\n  reproj err: " << err << "\n\n";
    }

    // ---- Step 1: Intrinsics ----
    std::cout << "--- Step 1: detect_and_calibrate_intrinsics ---\n";
    std::map<std::string, CalibrationPipeline::CameraIntrinsics> intrinsics;
    std::string status;
    if (!CalibrationPipeline::detect_and_calibrate_intrinsics(config, intrinsics, &status)) {
        std::cerr << "Intrinsics failed: " << status << "\n";
        return 1;
    }
    std::cout << status << "\n";
    for (const auto &[serial, intr] : intrinsics) {
        std::cout << "  " << serial << ": " << intr.image_width << "x"
                  << intr.image_height
                  << ", reproj=" << intr.reproj_error
                  << ", images=" << intr.corners_per_image.size() << "\n";
    }
    std::cout << "\n";

    // ---- Step 2: Build landmarks ----
    std::cout << "--- Step 2: build_landmarks ---\n";
    std::map<std::string, std::map<int, Eigen::Vector2d>> landmarks;
    CalibrationPipeline::build_landmarks(config, intrinsics, landmarks);
    for (const auto &[serial, lm] : landmarks) {
        std::cout << "  " << serial << ": " << lm.size() << " landmarks\n";
    }
    std::cout << "\n";

    // ---- Verify our target cameras exist ----
    if (!intrinsics.count(SERIAL_A) || !intrinsics.count(SERIAL_B)) {
        std::cerr << "Target camera serial not found in intrinsics!\n";
        return 1;
    }
    if (!landmarks.count(SERIAL_A) || !landmarks.count(SERIAL_B)) {
        std::cerr << "Target camera serial not found in landmarks!\n";
        return 1;
    }

    const auto &intr_a = intrinsics.at(SERIAL_A);
    const auto &intr_b = intrinsics.at(SERIAL_B);
    const auto &lm_a = landmarks.at(SERIAL_A);
    const auto &lm_b = landmarks.at(SERIAL_B);

    // ---- K matrices ----
    std::cout << "--- K matrices ---\n";
    print_eigen3("K_" + std::string(SERIAL_A), intr_a.K);
    std::cout << "  dist: " << intr_a.dist.transpose() << "\n";
    print_eigen3("K_" + std::string(SERIAL_B), intr_b.K);
    std::cout << "  dist: " << intr_b.dist.transpose() << "\n\n";

    // ---- Common landmarks ----
    std::vector<int> common_ids;
    for (const auto &[id, _] : lm_a) {
        if (lm_b.count(id))
            common_ids.push_back(id);
    }
    std::cout << "--- Common landmarks: " << common_ids.size() << " ---\n";
    if (common_ids.empty()) {
        std::cerr << "No common landmarks! Cannot proceed.\n";
        return 1;
    }

    // ---- Undistort points ----
    std::vector<cv::Point2f> pts_a_cv, pts_b_cv;
    std::vector<cv::Point2f> pts_a_norm, pts_b_norm;
    for (int id : common_ids) {
        Eigen::Vector2d ua = red_math::undistortPoint(lm_a.at(id), intr_a.K, intr_a.dist);
        Eigen::Vector2d ub = red_math::undistortPoint(lm_b.at(id), intr_b.K, intr_b.dist);
        pts_a_cv.push_back(cv::Point2f((float)ua.x(), (float)ua.y()));
        pts_b_cv.push_back(cv::Point2f((float)ub.x(), (float)ub.y()));

        // Normalized coords
        double fx_a = intr_a.K(0, 0), fy_a = intr_a.K(1, 1);
        double cx_a = intr_a.K(0, 2), cy_a = intr_a.K(1, 2);
        double fx_b = intr_b.K(0, 0), fy_b = intr_b.K(1, 1);
        double cx_b = intr_b.K(0, 2), cy_b = intr_b.K(1, 2);
        pts_a_norm.push_back(cv::Point2f(
            (float)((ua.x() - cx_a) / fx_a),
            (float)((ua.y() - cy_a) / fy_a)));
        pts_b_norm.push_back(cv::Point2f(
            (float)((ub.x() - cx_b) / fx_b),
            (float)((ub.y() - cy_b) / fy_b)));
    }

    // Print first 10 undistorted point pairs
    int n_print = std::min((int)common_ids.size(), 10);
    std::cout << "\nFirst " << n_print << " undistorted point pairs (pixel coords):\n";
    std::cout << std::fixed << std::setprecision(3);
    for (int i = 0; i < n_print; i++) {
        std::cout << "  id=" << std::setw(6) << common_ids[i]
                  << "  A=(" << std::setw(9) << pts_a_cv[i].x << ", "
                  << std::setw(9) << pts_a_cv[i].y << ")"
                  << "  B=(" << std::setw(9) << pts_b_cv[i].x << ", "
                  << std::setw(9) << pts_b_cv[i].y << ")\n";
    }
    std::cout << "\nFirst " << n_print << " normalized point pairs:\n";
    for (int i = 0; i < n_print; i++) {
        std::cout << "  id=" << std::setw(6) << common_ids[i]
                  << "  A=(" << std::setw(10) << std::setprecision(6)
                  << pts_a_norm[i].x << ", " << std::setw(10) << pts_a_norm[i].y << ")"
                  << "  B=(" << std::setw(10) << pts_b_norm[i].x << ", "
                  << std::setw(10) << pts_b_norm[i].y << ")\n";
    }

    // ---- Point spread statistics ----
    {
        float min_ax = 1e9, max_ax = -1e9, min_ay = 1e9, max_ay = -1e9;
        float min_bx = 1e9, max_bx = -1e9, min_by = 1e9, max_by = -1e9;
        for (size_t i = 0; i < pts_a_cv.size(); i++) {
            min_ax = std::min(min_ax, pts_a_cv[i].x);
            max_ax = std::max(max_ax, pts_a_cv[i].x);
            min_ay = std::min(min_ay, pts_a_cv[i].y);
            max_ay = std::max(max_ay, pts_a_cv[i].y);
            min_bx = std::min(min_bx, pts_b_cv[i].x);
            max_bx = std::max(max_bx, pts_b_cv[i].x);
            min_by = std::min(min_by, pts_b_cv[i].y);
            max_by = std::max(max_by, pts_b_cv[i].y);
        }
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\nPoint spread (undistorted pixel coords):\n";
        std::cout << "  A: x=[" << min_ax << ", " << max_ax << "]"
                  << "  y=[" << min_ay << ", " << max_ay << "]\n";
        std::cout << "  B: x=[" << min_bx << ", " << max_bx << "]"
                  << "  y=[" << min_by << ", " << max_by << "]\n";

        float min_nx = 1e9, max_nx = -1e9, min_ny = 1e9, max_ny = -1e9;
        for (size_t i = 0; i < pts_a_norm.size(); i++) {
            min_nx = std::min(min_nx, pts_a_norm[i].x);
            max_nx = std::max(max_nx, pts_a_norm[i].x);
            min_ny = std::min(min_ny, pts_a_norm[i].y);
            max_ny = std::max(max_ny, pts_a_norm[i].y);
        }
        std::cout << "  A_norm: x=[" << min_nx << ", " << max_nx << "]"
                  << "  y=[" << min_ny << ", " << max_ny << "]\n\n";
    }

    // ---- OpenCV K matrices (for findEssentialMat with pixel coords) ----
    cv::Mat Ka_cv, Kb_cv;
    cv::eigen2cv(intr_a.K, Ka_cv);
    cv::eigen2cv(intr_b.K, Kb_cv);

    // ====================================================================
    // Test 1: findFundamentalMat with pixel coords
    // ====================================================================
    std::cout << "=== Test 1: findFundamentalMat(FM_RANSAC, thresh=3.0) ===\n";
    {
        cv::Mat inlier_mask;
        cv::Mat F = cv::findFundamentalMat(pts_a_cv, pts_b_cv,
                                           cv::FM_RANSAC, 3.0, 0.999,
                                           inlier_mask);
        std::cout << "  F size: " << F.rows << "x" << F.cols << "\n";
        if (!F.empty() && F.rows == 3 && F.cols == 3) {
            print_matrix("  F", F);
            int inliers = cv::countNonZero(inlier_mask);
            std::cout << "  Inliers: " << inliers << " / "
                      << (int)pts_a_cv.size() << "\n";

            // Derive E = Kb^T * F * Ka
            cv::Mat E_from_F = Kb_cv.t() * F * Ka_cv;
            print_matrix("  E (from Kb^T * F * Ka)", E_from_F);

            // SVD of E to check singular values (should be [s, s, 0])
            cv::Mat w, u, vt;
            cv::SVD::compute(E_from_F, w, u, vt);
            std::cout << "  E singular values: " << w.at<double>(0) << ", "
                      << w.at<double>(1) << ", " << w.at<double>(2) << "\n";
        } else {
            std::cout << "  FAILED (returned " << F.rows << "x" << F.cols << ")\n";
            if (F.rows > 3)
                std::cout << "  (multiple solutions returned - degenerate config)\n";
        }
    }
    std::cout << "\n";

    // ====================================================================
    // Test 2: findEssentialMat with normalized coords + identity K
    // ====================================================================
    std::cout << "=== Test 2: findEssentialMat(normalized coords, K=I) ===\n";
    {
        cv::Mat inlier_mask;
        cv::Mat E = cv::findEssentialMat(pts_a_norm, pts_b_norm,
                                         cv::Mat::eye(3, 3, CV_64F),
                                         cv::RANSAC, 0.999, 0.001,
                                         inlier_mask);
        std::cout << "  E size: " << E.rows << "x" << E.cols << "\n";
        if (!E.empty() && E.rows == 3 && E.cols == 3) {
            print_matrix("  E", E);
            int inliers = cv::countNonZero(inlier_mask);
            std::cout << "  Inliers: " << inliers << " / "
                      << (int)pts_a_norm.size() << "\n";

            cv::Mat w, u, vt;
            cv::SVD::compute(E, w, u, vt);
            std::cout << "  E singular values: " << w.at<double>(0) << ", "
                      << w.at<double>(1) << ", " << w.at<double>(2) << "\n";
        } else {
            std::cout << "  FAILED (returned " << E.rows << "x" << E.cols << ")\n";
            if (E.rows > 3)
                std::cout << "  (multiple solutions returned - degenerate config)\n";
        }
    }
    std::cout << "\n";

    // ====================================================================
    // Test 3: findEssentialMat with pixel coords + actual K
    // ====================================================================
    std::cout << "=== Test 3: findEssentialMat(pixel coords, K=Ka) ===\n";
    std::cout << "  Note: uses Ka for both cameras (OpenCV 1-camera overload)\n";
    {
        cv::Mat inlier_mask;
        cv::Mat E = cv::findEssentialMat(pts_a_cv, pts_b_cv,
                                         Ka_cv, cv::RANSAC, 0.999, 1.0,
                                         inlier_mask);
        std::cout << "  E size: " << E.rows << "x" << E.cols << "\n";
        if (!E.empty() && E.rows == 3 && E.cols == 3) {
            print_matrix("  E", E);
            int inliers = cv::countNonZero(inlier_mask);
            std::cout << "  Inliers: " << inliers << " / "
                      << (int)pts_a_cv.size() << "\n";

            cv::Mat w, u, vt;
            cv::SVD::compute(E, w, u, vt);
            std::cout << "  E singular values: " << w.at<double>(0) << ", "
                      << w.at<double>(1) << ", " << w.at<double>(2) << "\n";
        } else {
            std::cout << "  FAILED (returned " << E.rows << "x" << E.cols << ")\n";
            if (E.rows > 3)
                std::cout << "  (multiple solutions returned - degenerate config)\n";
        }
    }
    std::cout << "\n";

    // ====================================================================
    // Test 4: findEssentialMat with pixel coords + separate Ka, Kb
    // ====================================================================
    std::cout << "=== Test 4: findEssentialMat(pixel coords, Ka, Kb) ===\n";
    {
        cv::Mat inlier_mask;
        // OpenCV 4.x 2-camera overload: findEssentialMat(p1, p2, Ka, Da, Kb, Db, ...)
        cv::Mat Da, Db; // empty = zero distortion (already undistorted)
        cv::Mat E = cv::findEssentialMat(pts_a_cv, pts_b_cv,
                                         Ka_cv, Da, Kb_cv, Db,
                                         cv::RANSAC, 0.999, 1.0,
                                         inlier_mask);
        std::cout << "  E size: " << E.rows << "x" << E.cols << "\n";
        if (!E.empty() && E.rows == 3 && E.cols == 3) {
            print_matrix("  E", E);
            int inliers = cv::countNonZero(inlier_mask);
            std::cout << "  Inliers: " << inliers << " / "
                      << (int)pts_a_cv.size() << "\n";

            cv::Mat w, u, vt;
            cv::SVD::compute(E, w, u, vt);
            std::cout << "  E singular values: " << w.at<double>(0) << ", "
                      << w.at<double>(1) << ", " << w.at<double>(2) << "\n";
        } else {
            std::cout << "  FAILED (returned " << E.rows << "x" << E.cols << ")\n";
            if (E.rows > 3)
                std::cout << "  (multiple solutions returned - degenerate config)\n";
        }
    }
    std::cout << "\n";

    // ====================================================================
    // Test 5: Vary RANSAC thresholds for the normalized-coord path
    // ====================================================================
    std::cout << "=== Test 5: findEssentialMat normalized — sweep thresholds ===\n";
    for (double thresh : {0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05}) {
        cv::Mat inlier_mask;
        cv::Mat E = cv::findEssentialMat(pts_a_norm, pts_b_norm,
                                         cv::Mat::eye(3, 3, CV_64F),
                                         cv::RANSAC, 0.999, thresh,
                                         inlier_mask);
        int inliers = (!inlier_mask.empty()) ? cv::countNonZero(inlier_mask) : 0;
        std::cout << "  thresh=" << std::setw(8) << std::setprecision(4) << thresh
                  << "  E_size=" << E.rows << "x" << E.cols
                  << "  inliers=" << inliers << "/" << (int)pts_a_norm.size();
        if (!E.empty() && E.rows == 3 && E.cols == 3) {
            cv::Mat w, u, vt;
            cv::SVD::compute(E, w, u, vt);
            std::cout << "  sv=(" << std::setprecision(6)
                      << w.at<double>(0) << ", " << w.at<double>(1) << ", "
                      << w.at<double>(2) << ")";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // ====================================================================
    // Test 6: recoverPose from the best E (if Test 2 succeeded)
    // ====================================================================
    std::cout << "=== Test 6: recoverPose (from Test 2 E if available) ===\n";
    {
        cv::Mat inlier_mask;
        cv::Mat E = cv::findEssentialMat(pts_a_norm, pts_b_norm,
                                         cv::Mat::eye(3, 3, CV_64F),
                                         cv::RANSAC, 0.999, 0.001,
                                         inlier_mask);
        if (!E.empty() && E.rows == 3 && E.cols == 3) {
            cv::Mat R_cv, t_cv;
            int good = cv::recoverPose(E, pts_a_norm, pts_b_norm,
                                       cv::Mat::eye(3, 3, CV_64F),
                                       R_cv, t_cv, inlier_mask);
            std::cout << "  recoverPose: " << good << " points in front\n";
            print_matrix("  R", R_cv);
            print_matrix("  t", t_cv);
        } else {
            std::cout << "  Skipped (E estimation failed)\n";
        }
    }
    std::cout << "\n";

    std::cout << "=== Done ===\n";
    return 0;
}
