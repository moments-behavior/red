// eval_calibration.cpp — Evaluate calibration quality with global consistency metric.
// Loads calibration YAMLs + landmarks.json, computes per-board and multi-view reproj.
// No BA, no refinement — pure evaluation.

#include "calibration_pipeline.h"
#include "calibration_tool.h"
#include "opencv_yaml_io.h"
#include "red_math.h"
#include "json.hpp"
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>
#include <map>
#include <set>

namespace fs = std::filesystem;

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <calib_folder> <landmarks.json> [--name TAG]\n", argv[0]);
        printf("  Evaluates calibration quality using global multi-view triangulation.\n");
        return 1;
    }

    std::string calib_folder = argv[1];
    std::string landmarks_file = argv[2];
    std::string name = "Calibration";
    std::string ba_points_override;
    for (int i = 3; i < argc; i++) {
        if (std::string(argv[i]) == "--name" && i + 1 < argc) name = argv[++i];
        else if (std::string(argv[i]) == "--ba-points" && i + 1 < argc) ba_points_override = argv[++i];
    }

    // Discover cameras from YAML files
    std::vector<std::string> cam_names;
    for (const auto &entry : fs::directory_iterator(calib_folder)) {
        if (entry.path().extension() != ".yaml") continue;
        std::string fname = entry.path().stem().string();
        if (fname.substr(0, 3) == "Cam")
            cam_names.push_back(fname.substr(3));
    }
    std::sort(cam_names.begin(), cam_names.end());

    // Load calibration
    int nc = (int)cam_names.size();
    std::vector<CalibrationPipeline::CameraPose> poses(nc);
    for (int c = 0; c < nc; c++) {
        std::string yaml_path = calib_folder + "/Cam" + cam_names[c] + ".yaml";
        auto yaml = opencv_yaml::read(yaml_path);
        poses[c].K = yaml.getMatrix("camera_matrix").block<3, 3>(0, 0);
        Eigen::MatrixXd dist_mat = yaml.getMatrix("distortion_coefficients");
        for (int j = 0; j < 5; j++) poses[c].dist(j) = dist_mat(j, 0);
        poses[c].R = yaml.getMatrix("rc_ext").block<3, 3>(0, 0);
        Eigen::MatrixXd t_mat = yaml.getMatrix("tc_ext");
        poses[c].t = Eigen::Vector3d(t_mat(0, 0), t_mat(1, 0), t_mat(2, 0));
    }

    // Load landmarks
    std::map<std::string, std::map<int, Eigen::Vector2d>> landmarks;
    {
        std::ifstream f(landmarks_file);
        nlohmann::json j;
        f >> j;
        for (auto &[cam, cam_j] : j.items()) {
            auto &ids = cam_j["ids"];
            auto &pts = cam_j["landmarks"];
            for (int i = 0; i < (int)ids.size(); i++) {
                landmarks[cam][ids[i].get<int>()] =
                    Eigen::Vector2d(pts[i][0].get<double>(), pts[i][1].get<double>());
            }
        }
    }

    // Count stats
    int total_obs = 0;
    std::set<int> all_ids;
    int cams_with_landmarks = 0;
    for (const auto &[cam, pts] : landmarks) {
        total_obs += (int)pts.size();
        if (!pts.empty()) cams_with_landmarks++;
        for (const auto &[id, _] : pts) all_ids.insert(id);
    }

    printf("\n========================================\n");
    printf("  %s\n", name.c_str());
    printf("========================================\n\n");
    printf("Calib folder:  %s\n", calib_folder.c_str());
    printf("Landmarks:     %s\n", landmarks_file.c_str());
    printf("Cameras:       %d (with landmarks: %d)\n", nc, cams_with_landmarks);
    printf("Landmarks:     %d unique, %d observations\n", (int)all_ids.size(), total_obs);
    printf("\n");

    // Compute per-board reproj (project using known 3D points from BA)
    std::string ba_pts_path;
    if (!ba_points_override.empty()) {
        ba_pts_path = ba_points_override;
    } else {
        ba_pts_path = fs::path(landmarks_file).parent_path().string() + "/bundle_adjustment/ba_points.json";
        if (!fs::exists(ba_pts_path))
            ba_pts_path = fs::path(landmarks_file).parent_path().string() + "/ba_points.json";
    }

    if (fs::exists(ba_pts_path)) {
        std::map<int, Eigen::Vector3d> ba_points;
        std::ifstream pf(ba_pts_path);
        nlohmann::json pj;
        pf >> pj;

        // Handle both formats:
        // RED: {"0": [x,y,z], "1": [x,y,z], ...}
        // MVC: {"points_3d": [[x,y,z], ...], "ids": [0, 1, ...]}
        if (pj.contains("points_3d") && pj.contains("ids")) {
            const auto &pts = pj["points_3d"];
            const auto &ids = pj["ids"];
            for (int i = 0; i < (int)ids.size(); i++)
                ba_points[ids[i].get<int>()] = Eigen::Vector3d(
                    pts[i][0].get<double>(), pts[i][1].get<double>(), pts[i][2].get<double>());
        } else {
            for (auto &[id_str, pt] : pj.items())
                ba_points[std::stoi(id_str)] = Eigen::Vector3d(
                    pt[0].get<double>(), pt[1].get<double>(), pt[2].get<double>());
        }

        // Compute per-board reproj using BA 3D points
        std::vector<double> per_cam_mean(nc, 0.0);
        std::vector<int> per_cam_count(nc, 0);
        double total_err = 0;
        int total_count = 0;

        for (int c = 0; c < nc; c++) {
            auto it = landmarks.find(cam_names[c]);
            if (it == landmarks.end()) continue;
            for (const auto &[pid, px] : it->second) {
                auto pit = ba_points.find(pid);
                if (pit == ba_points.end()) continue;
                double e = (red_math::projectPointR(pit->second, poses[c].R, poses[c].t,
                                                      poses[c].K, poses[c].dist) - px).norm();
                per_cam_mean[c] += e;
                per_cam_count[c]++;
                total_err += e;
                total_count++;
            }
        }

        double mean_reproj = total_count > 0 ? total_err / total_count : 0;
        printf("--- Per-board reprojection (using BA 3D points) ---\n");
        printf("Mean:  %.3f px (%d observations)\n\n", mean_reproj, total_count);

        printf("%-12s %6s %8s\n", "Camera", "Obs", "Mean(px)");
        printf("%-12s %6s %8s\n", "------", "---", "--------");
        for (int c = 0; c < nc; c++) {
            if (per_cam_count[c] == 0) continue;
            printf("Cam%-9s %6d %8.3f\n", cam_names[c].c_str(), per_cam_count[c],
                   per_cam_mean[c] / per_cam_count[c]);
        }
        printf("\n");
    }

    // Compute global multi-view triangulation consistency
    auto gc = CalibrationPipeline::compute_global_consistency(landmarks, poses, cam_names);

    printf("--- Multi-view triangulation consistency ---\n");
    printf("Mean:   %.2f px\n", gc.mean_reproj);
    printf("Median: %.2f px\n", gc.median_reproj);
    printf("95pct:  %.2f px\n", gc.pct95_reproj);
    printf("Landmarks triangulated: %d\n", gc.landmarks_triangulated);
    printf("Total observations:     %d\n\n", gc.total_observations);

    printf("%-12s %6s %8s\n", "Camera", "Obs", "Mean(px)");
    printf("%-12s %6s %8s\n", "------", "---", "--------");
    for (const auto &cr : gc.per_camera) {
        if (cr.obs == 0) continue;
        printf("Cam%-9s %6d %8.2f\n", cr.name.c_str(), cr.obs, cr.mean_reproj);
    }

    // ---- Known-geometry validation ----
    // ChArUco board corners have known spacing. Triangulate pairs of adjacent
    // corners and compare 3D distance to the known square size.
    // Landmark IDs encode position on the board: id = row * cols + col
    // (where cols = squares_x - 1 for inner corners of a ChArUco board).
    // Adjacent corners (horizontally or vertically) should be exactly square_size apart.
    {
        // Determine board layout from landmark IDs
        // For a 5x5 ChArUco board (5 squares wide, 5 tall), inner corners are 4x4 = 16 per board.
        // But the actual layout depends on the detection. Let's just check all pairs of landmarks
        // that are triangulated and compute pairwise 3D distances to find the known spacing.

        // First triangulate all landmarks
        CalibrationTool::CalibConfig cc;
        cc.cam_ordered = cam_names;
        std::map<int, Eigen::Vector3d> tri_pts;
        CalibrationPipeline::triangulate_landmarks_multiview(cc, landmarks, poses, tri_pts, 50.0);

        if (tri_pts.size() >= 2) {
            // Compute all pairwise distances
            std::vector<double> all_dists;
            std::vector<int> ids;
            for (const auto &[id, _] : tri_pts) ids.push_back(id);

            for (int i = 0; i < (int)ids.size(); i++) {
                for (int j = i + 1; j < (int)ids.size(); j++) {
                    double d = (tri_pts[ids[i]] - tri_pts[ids[j]]).norm();
                    all_dists.push_back(d);
                }
            }

            // Find the mode of shortest distances — this should be the square size
            std::sort(all_dists.begin(), all_dists.end());

            // The known square size (80mm for this board)
            double known_square = 80.0; // TODO: pass via CLI
            printf("--- Known-geometry validation (square_size = %.1f mm) ---\n", known_square);

            // Find distances near the expected square size
            std::vector<double> near_square;
            for (double d : all_dists) {
                if (d > known_square * 0.5 && d < known_square * 1.5)
                    near_square.push_back(d);
            }

            // Also check diagonal (sqrt(2) * square_size)
            double known_diag = known_square * std::sqrt(2.0);
            std::vector<double> near_diag;
            for (double d : all_dists) {
                if (d > known_diag * 0.8 && d < known_diag * 1.2)
                    near_diag.push_back(d);
            }

            if (!near_square.empty()) {
                double sum = 0;
                for (double d : near_square) sum += d;
                double mean_dist = sum / near_square.size();
                double max_err = 0;
                std::vector<double> errs;
                for (double d : near_square) {
                    double e = std::abs(d - known_square);
                    errs.push_back(e);
                    if (e > max_err) max_err = e;
                }
                std::sort(errs.begin(), errs.end());
                double mean_err = 0;
                for (double e : errs) mean_err += e;
                mean_err /= errs.size();

                printf("Adjacent corners (expected %.1f mm):\n", known_square);
                printf("  Count:      %d pairs\n", (int)near_square.size());
                printf("  Mean dist:  %.2f mm (error: %.2f mm, %.2f%%)\n",
                       mean_dist, std::abs(mean_dist - known_square),
                       std::abs(mean_dist - known_square) / known_square * 100);
                printf("  Mean error: %.2f mm\n", mean_err);
                printf("  Median err: %.2f mm\n", errs[errs.size() / 2]);
                printf("  Max error:  %.2f mm\n", max_err);
            }

            if (!near_diag.empty()) {
                double sum = 0;
                for (double d : near_diag) sum += d;
                double mean_dist = sum / near_diag.size();
                std::vector<double> errs;
                for (double d : near_diag) errs.push_back(std::abs(d - known_diag));
                std::sort(errs.begin(), errs.end());
                double mean_err = 0;
                for (double e : errs) mean_err += e;
                mean_err /= errs.size();

                printf("Diagonal corners (expected %.1f mm):\n", known_diag);
                printf("  Count:      %d pairs\n", (int)near_diag.size());
                printf("  Mean dist:  %.2f mm (error: %.2f mm, %.2f%%)\n",
                       mean_dist, std::abs(mean_dist - known_diag),
                       std::abs(mean_dist - known_diag) / known_diag * 100);
                printf("  Mean error: %.2f mm\n", mean_err);
                printf("  Median err: %.2f mm\n", errs[errs.size() / 2]);
            }

            printf("  Total triangulated: %d landmarks\n", (int)tri_pts.size());
        }
    }

    printf("\n========================================\n\n");
    return 0;
}
