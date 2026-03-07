// test_calib_debug.cpp — Standalone diagnostic for failing camera pair.
// Loads config, runs Steps 1-2 of the calibration pipeline, then exercises
// findEssentialMatRANSAC / findFundamentalMatRANSAC on the pair (2006054 -> 2002486)
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
        auto aruco_dict = aruco_detect::getDictionary(cs.dictionary);
        aruco_detect::CharucoBoard board;
        board.squares_x = cs.w;
        board.squares_y = cs.h;
        board.square_length = cs.square_side_length;
        board.marker_length = cs.marker_side_length;
        board.dictionary_id = cs.dictionary;

        auto img_nums = CalibrationPipeline::get_sorted_image_numbers(
            config.img_path, SERIAL_A);
        auto ext = CalibrationPipeline::get_image_extension(
            config.img_path, SERIAL_A);

        std::vector<std::vector<Eigen::Vector3f>> all_obj;
        std::vector<std::vector<Eigen::Vector2f>> all_img;
        int im_w = 0, im_h = 0;

        for (int n : img_nums) {
            std::string f = config.img_path + "/" + std::string(SERIAL_A) +
                            "_" + std::to_string(n) + ext;
            int w = 0, h = 0, ch = 0;
            unsigned char *px = stbi_load(f.c_str(), &w, &h, &ch, 1);
            if (!px) continue;
            if (im_w == 0) { im_w = w; im_h = h; }

            auto charuco = aruco_detect::detectCharucoBoard(
                px, w, h, board, aruco_dict);
            if ((int)charuco.ids.size() < 6) { stbi_image_free(px); continue; }

            aruco_detect::cornerSubPix(px, w, h, charuco.corners, 3, 30, 0.01f);

            std::vector<Eigen::Vector3f> obj_pts;
            std::vector<Eigen::Vector2f> img_pts;
            aruco_detect::matchImagePoints(
                board, charuco.corners, charuco.ids, obj_pts, img_pts);

            std::cout << "  img " << n << ": detectBoard=" << charuco.ids.size()
                      << " corners, matchImagePoints=" << obj_pts.size()
                      << " obj_pts, " << img_pts.size() << " img_pts\n";
            if ((int)obj_pts.size() >= 6) {
                all_obj.push_back(std::move(obj_pts));
                all_img.push_back(std::move(img_pts));
            }
            stbi_image_free(px);
            if (all_obj.size() >= 5) break; // just test a few
        }

        std::cout << "  Total views for calibration: " << all_obj.size() << "\n";
        std::cout << "  Image size: " << im_w << "x" << im_h << "\n";

        auto calib = intrinsic_calib::calibrateCamera(
            all_obj, all_img, im_w, im_h, /*fix_aspect_ratio=*/true);

        std::cout << "  K after:  fx=" << calib.K(0,0) << " fy="
                  << calib.K(1,1) << " cx=" << calib.K(0,2)
                  << " cy=" << calib.K(1,2) << "\n";
        std::cout << "  dist: " << calib.dist.transpose() << "\n";
        std::cout << "  reproj err: " << calib.reproj_error << "\n\n";
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

    // ---- Undistort points (Eigen vectors) ----
    std::vector<Eigen::Vector2d> pts_a_undist, pts_b_undist;
    std::vector<Eigen::Vector2d> pts_a_norm, pts_b_norm;
    for (int id : common_ids) {
        Eigen::Vector2d ua = red_math::undistortPoint(lm_a.at(id), intr_a.K, intr_a.dist);
        Eigen::Vector2d ub = red_math::undistortPoint(lm_b.at(id), intr_b.K, intr_b.dist);
        pts_a_undist.push_back(ua);
        pts_b_undist.push_back(ub);

        // Normalized coords
        double fx_a = intr_a.K(0, 0), fy_a = intr_a.K(1, 1);
        double cx_a = intr_a.K(0, 2), cy_a = intr_a.K(1, 2);
        double fx_b = intr_b.K(0, 0), fy_b = intr_b.K(1, 1);
        double cx_b = intr_b.K(0, 2), cy_b = intr_b.K(1, 2);
        pts_a_norm.push_back(Eigen::Vector2d(
            (ua.x() - cx_a) / fx_a, (ua.y() - cy_a) / fy_a));
        pts_b_norm.push_back(Eigen::Vector2d(
            (ub.x() - cx_b) / fx_b, (ub.y() - cy_b) / fy_b));
    }

    // Print first 10 undistorted point pairs
    int n_print = std::min((int)common_ids.size(), 10);
    std::cout << "\nFirst " << n_print << " undistorted point pairs (pixel coords):\n";
    std::cout << std::fixed << std::setprecision(3);
    for (int i = 0; i < n_print; i++) {
        std::cout << "  id=" << std::setw(6) << common_ids[i]
                  << "  A=(" << std::setw(9) << pts_a_undist[i].x() << ", "
                  << std::setw(9) << pts_a_undist[i].y() << ")"
                  << "  B=(" << std::setw(9) << pts_b_undist[i].x() << ", "
                  << std::setw(9) << pts_b_undist[i].y() << ")\n";
    }
    std::cout << "\nFirst " << n_print << " normalized point pairs:\n";
    for (int i = 0; i < n_print; i++) {
        std::cout << "  id=" << std::setw(6) << common_ids[i]
                  << "  A=(" << std::setw(10) << std::setprecision(6)
                  << pts_a_norm[i].x() << ", " << std::setw(10) << pts_a_norm[i].y() << ")"
                  << "  B=(" << std::setw(10) << pts_b_norm[i].x() << ", "
                  << std::setw(10) << pts_b_norm[i].y() << ")\n";
    }

    // ---- Point spread statistics ----
    {
        double min_ax = 1e9, max_ax = -1e9, min_ay = 1e9, max_ay = -1e9;
        double min_bx = 1e9, max_bx = -1e9, min_by = 1e9, max_by = -1e9;
        for (size_t i = 0; i < pts_a_undist.size(); i++) {
            min_ax = std::min(min_ax, pts_a_undist[i].x());
            max_ax = std::max(max_ax, pts_a_undist[i].x());
            min_ay = std::min(min_ay, pts_a_undist[i].y());
            max_ay = std::max(max_ay, pts_a_undist[i].y());
            min_bx = std::min(min_bx, pts_b_undist[i].x());
            max_bx = std::max(max_bx, pts_b_undist[i].x());
            min_by = std::min(min_by, pts_b_undist[i].y());
            max_by = std::max(max_by, pts_b_undist[i].y());
        }
        std::cout << std::fixed << std::setprecision(2);
        std::cout << "\nPoint spread (undistorted pixel coords):\n";
        std::cout << "  A: x=[" << min_ax << ", " << max_ax << "]"
                  << "  y=[" << min_ay << ", " << max_ay << "]\n";
        std::cout << "  B: x=[" << min_bx << ", " << max_bx << "]"
                  << "  y=[" << min_by << ", " << max_by << "]\n";

        double min_nx = 1e9, max_nx = -1e9, min_ny = 1e9, max_ny = -1e9;
        for (size_t i = 0; i < pts_a_norm.size(); i++) {
            min_nx = std::min(min_nx, pts_a_norm[i].x());
            max_nx = std::max(max_nx, pts_a_norm[i].x());
            min_ny = std::min(min_ny, pts_a_norm[i].y());
            max_ny = std::max(max_ny, pts_a_norm[i].y());
        }
        std::cout << "  A_norm: x=[" << min_nx << ", " << max_nx << "]"
                  << "  y=[" << min_ny << ", " << max_ny << "]\n\n";
    }

    // ====================================================================
    // Test 1: Eigen findFundamentalMatRANSAC with pixel coords
    // ====================================================================
    std::cout << "=== Test 1: red_math::findFundamentalMatRANSAC(thresh=3.0) ===\n";
    {
        auto result = red_math::findFundamentalMatRANSAC(
            pts_a_undist, pts_b_undist, 0.999, 3.0);
        if (result.success) {
            print_eigen3("  F", result.E);
            std::cout << "  Inliers: " << result.num_inliers << " / "
                      << (int)pts_a_undist.size() << "\n";

            // Derive E = Kb^T * F * Ka
            Eigen::Matrix3d E_from_F = intr_b.K.transpose() * result.E * intr_a.K;
            print_eigen3("  E (from Kb^T * F * Ka)", E_from_F);

            Eigen::JacobiSVD<Eigen::Matrix3d> svd(E_from_F, Eigen::ComputeFullU | Eigen::ComputeFullV);
            auto sv = svd.singularValues();
            std::cout << "  E singular values: " << sv(0) << ", "
                      << sv(1) << ", " << sv(2) << "\n";
        } else {
            std::cout << "  FAILED\n";
        }
    }
    std::cout << "\n";

    // ====================================================================
    // Test 2: Eigen findEssentialMatRANSAC with normalized coords
    // ====================================================================
    std::cout << "=== Test 2: red_math::findEssentialMatRANSAC(normalized coords) ===\n";
    {
        auto result = red_math::findEssentialMatRANSAC(
            pts_a_norm, pts_b_norm, 0.999, 0.001);
        if (result.success) {
            print_eigen3("  E", result.E);
            std::cout << "  Inliers: " << result.num_inliers << " / "
                      << (int)pts_a_norm.size() << "\n";

            Eigen::JacobiSVD<Eigen::Matrix3d> svd(result.E, Eigen::ComputeFullU | Eigen::ComputeFullV);
            auto sv = svd.singularValues();
            std::cout << "  E singular values: " << sv(0) << ", "
                      << sv(1) << ", " << sv(2) << "\n";
        } else {
            std::cout << "  FAILED\n";
        }
    }
    std::cout << "\n";

    // ====================================================================
    // Test 5: Vary RANSAC thresholds for the normalized-coord path
    // ====================================================================
    std::cout << "=== Test 5: findEssentialMatRANSAC normalized — sweep thresholds ===\n";
    for (double thresh : {0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05}) {
        auto result = red_math::findEssentialMatRANSAC(
            pts_a_norm, pts_b_norm, 0.999, thresh);
        std::cout << "  thresh=" << std::setw(8) << std::setprecision(4) << thresh
                  << "  inliers=" << result.num_inliers << "/" << (int)pts_a_norm.size();
        if (result.success) {
            Eigen::JacobiSVD<Eigen::Matrix3d> svd(result.E, Eigen::ComputeFullU | Eigen::ComputeFullV);
            auto sv = svd.singularValues();
            std::cout << "  sv=(" << std::setprecision(6)
                      << sv(0) << ", " << sv(1) << ", " << sv(2) << ")";
        }
        std::cout << "\n";
    }
    std::cout << "\n";

    // ====================================================================
    // Test 6: decomposeEssentialMatrix + chiral test (replaces recoverPose)
    // ====================================================================
    std::cout << "=== Test 6: decomposeEssentialMatrix + chiral test ===\n";
    {
        auto e_result = red_math::findEssentialMatRANSAC(
            pts_a_norm, pts_b_norm, 0.999, 0.001);
        if (e_result.success) {
            auto decomp = red_math::decomposeEssentialMatrix(e_result.E);
            print_eigen3("  R1", decomp.R1);
            print_eigen3("  R2", decomp.R2);
            std::cout << "  t: " << decomp.t.transpose() << "\n";

            // Test all 4 candidates to find the one with most points in front
            struct Candidate { Eigen::Matrix3d R; Eigen::Vector3d t; };
            Candidate cands[4] = {
                {decomp.R1,  decomp.t}, {decomp.R1, -decomp.t},
                {decomp.R2,  decomp.t}, {decomp.R2, -decomp.t}};

            int best_idx = 0, best_count = 0;
            for (int c = 0; c < 4; c++) {
                auto P_a = red_math::projectionFromKRt(
                    Eigen::Matrix3d::Identity(),
                    Eigen::Matrix3d::Identity(),
                    Eigen::Vector3d::Zero());
                auto P_b = red_math::projectionFromKRt(
                    Eigen::Matrix3d::Identity(), cands[c].R, cands[c].t);

                int positive = 0;
                for (int i = 0; i < (int)common_ids.size(); i++) {
                    if (!e_result.inlier_mask[i]) continue;
                    Eigen::Vector3d X = red_math::triangulatePoints(
                        {pts_a_norm[i], pts_b_norm[i]}, {P_a, P_b});
                    Eigen::Vector3d X_b = cands[c].R * X + cands[c].t;
                    if (X.z() > 0 && X_b.z() > 0) positive++;
                }
                if (positive > best_count) {
                    best_count = positive;
                    best_idx = c;
                }
            }

            std::cout << "  Best candidate: " << best_idx
                      << " (" << best_count << " points in front)\n";
            print_eigen3("  R", cands[best_idx].R);
            std::cout << "  t: " << cands[best_idx].t.transpose() << "\n";
        } else {
            std::cout << "  Skipped (E estimation failed)\n";
        }
    }
    std::cout << "\n";

    std::cout << "=== Done ===\n";
    return 0;
}
