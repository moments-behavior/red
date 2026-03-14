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
        printf("Usage: %s <landmarks.json> <calib_folder> [output_folder] [rot_prior] [trans_prior]\n", argv[0]);
        printf("\n  landmarks.json:  from data_exporter/calibration_refinement.py\n");
        printf("  calib_folder:    folder with Cam*.yaml files\n");
        printf("  output_folder:   where to write refined YAMLs (default: calib_folder/feature_refined)\n");
        printf("  rot_prior:       rotation prior weight (default: 10.0)\n");
        printf("  trans_prior:     translation prior weight (default: 100.0)\n");
        return 1;
    }

    std::string landmarks_file = argv[1];
    std::string calib_folder = argv[2];
    std::string output_folder = argc > 3 ? argv[3] : "";
    double rot_prior = argc > 4 ? std::atof(argv[4]) : 10.0;
    double trans_prior = argc > 5 ? std::atof(argv[5]) : 100.0;

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
    config.prior_rot_weight = rot_prior;
    config.prior_trans_weight = trans_prior;
    printf("Prior weights: rot=%.1f, trans=%.1f\n", rot_prior, trans_prior);

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
