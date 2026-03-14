// test_feature_refinement.cpp — Run feature-based calibration refinement
// from landmarks.json produced by data_exporter/calibration_refinement.py
//
// Build: cmake target "test_feature_refinement"
// Run:   ./test_feature_refinement <landmarks.json> <calib_folder> [output_folder]

#include "feature_refinement.h"

// Logger required by FFmpegDemuxer (NvCodec logging infrastructure)
#include "Logger.h"
simplelogger::Logger *logger =
    simplelogger::LoggerFactory::CreateConsoleLogger();

#include <cstdio>
#include <filesystem>

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <landmarks.json> <calib_folder> [output_folder]\n", argv[0]);
        printf("\n  landmarks.json: from data_exporter/calibration_refinement.py\n");
        printf("  calib_folder:   folder with Cam*.yaml files (init_calibration)\n");
        printf("  output_folder:  where to write refined YAMLs (default: calib_folder/feature_refined)\n");
        return 1;
    }

    std::string landmarks_file = argv[1];
    std::string calib_folder = argv[2];
    std::string output_folder = argc > 3 ? argv[3] : "";

    // Discover camera names from YAML files in calib_folder
    namespace fs = std::filesystem;
    std::vector<std::string> camera_names;
    for (const auto &entry : fs::directory_iterator(calib_folder)) {
        std::string name = entry.path().filename().string();
        if (name.size() > 7 && name.substr(0, 3) == "Cam" && name.substr(name.size() - 5) == ".yaml") {
            camera_names.push_back(name.substr(3, name.size() - 8));
        }
    }
    std::sort(camera_names.begin(), camera_names.end());
    printf("Found %d cameras in %s\n", (int)camera_names.size(), calib_folder.c_str());
    for (const auto &name : camera_names)
        printf("  Cam%s\n", name.c_str());

    // Derive points_3d.json path from landmarks.json path
    std::string points_3d_file;
    {
        auto pos = landmarks_file.rfind('/');
        if (pos != std::string::npos)
            points_3d_file = landmarks_file.substr(0, pos) + "/points_3d.json";
    }

    // Configure
    FeatureRefinement::FeatureConfig config;
    config.landmarks_file = landmarks_file;
    config.points_3d_file = points_3d_file;
    config.calibration_folder = calib_folder;
    config.output_folder = output_folder;
    config.camera_names = camera_names;
    config.reproj_threshold = 20.0;   // moderate — Python pre-triangulates good points
    config.ba_outlier_th1 = 10.0;
    config.ba_outlier_th2 = 3.0;
    config.ba_max_iter = 100;
    config.lock_intrinsics = true;
    config.lock_distortion = true;

    // Quick debug: manually check a few landmarks
    {
        nlohmann::json j;
        std::ifstream f(landmarks_file);
        f >> j;
        std::map<std::string, std::map<int, Eigen::Vector2d>> landmarks;
        for (auto &[cam, cam_j] : j.items()) {
            for (int i = 0; i < (int)cam_j["ids"].size(); i++)
                landmarks[cam][cam_j["ids"][i].get<int>()] = Eigen::Vector2d(
                    cam_j["landmarks"][i][0].get<double>(), cam_j["landmarks"][i][1].get<double>());
        }
        // Count points with >= 2 cameras
        std::map<int, std::vector<std::string>> point_cams;
        for (auto &[cam, pts] : landmarks)
            for (auto &[pid, _] : pts)
                point_cams[pid].push_back(cam);
        int multi = 0;
        for (auto &[pid, cams] : point_cams)
            if (cams.size() >= 2) multi++;
        printf("DEBUG: %d points in landmarks, %d with >= 2 cameras\n",
               (int)point_cams.size(), multi);

        // Check if cam_ordered matches
        printf("DEBUG: cam_ordered = [");
        for (const auto &n : config.camera_names) printf("%s, ", n.c_str());
        printf("]\n");

        // Try manual triangulation of first multi-view point
        for (auto &[pid, cams] : point_cams) {
            if (cams.size() < 2) continue;
            printf("DEBUG: point %d seen in %d cameras: ", pid, (int)cams.size());
            for (auto &c : cams) printf("%s ", c.c_str());
            printf("\n");
            // Check if these cameras are in cam_ordered
            for (auto &c : cams) {
                bool found = false;
                for (int i = 0; i < (int)config.camera_names.size(); i++) {
                    if (config.camera_names[i] == c) { found = true; break; }
                }
                if (!found) printf("  WARNING: camera %s not in cam_ordered!\n", c.c_str());
            }
            break;  // just first point
        }
    }

    // Run
    std::string status;
    auto result = FeatureRefinement::run_feature_refinement(config, &status);

    if (!result.success) {
        printf("\nFAILED: %s\n", result.error.c_str());
        return 1;
    }

    printf("\n=== Feature Refinement Result ===\n");
    printf("  Tracks:      %d\n", result.total_tracks);
    printf("  3D points:   %d\n", result.valid_3d_points);
    printf("  Observations: %d\n", result.total_observations);
    printf("  Outliers:    %d\n", result.ba_outliers_removed);
    printf("  Reproj:      %.3f px -> %.3f px\n", result.mean_reproj_before, result.mean_reproj_after);
    printf("  Output:      %s\n", result.output_folder.c_str());

    return 0;
}
