// test_holdout_compare.cpp — Run holdout cross-validation on existing landmarks.json
// This lets us test whether Python LightGlue matches produce genuine calibration improvement.

#include "feature_refinement.h"
#include <cstdio>
#include <filesystem>
#include <string>

namespace fs = std::filesystem;

int main(int argc, char **argv) {
    if (argc < 3) {
        printf("Usage: %s <landmarks.json> <calib_folder> [--rot-prior F] [--trans-prior F]\n", argv[0]);
        return 1;
    }

    std::string landmarks_file = argv[1];
    std::string calib_folder = argv[2];
    float rot_prior = 50.0f;
    float trans_prior = 500.0f;
    int holdout_seed = 42;
    float outlier_th1 = 10.0f;
    float outlier_th2 = 3.0f;

    for (int i = 3; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--rot-prior" && i + 1 < argc) rot_prior = std::stof(argv[++i]);
        else if (arg == "--trans-prior" && i + 1 < argc) trans_prior = std::stof(argv[++i]);
        else if (arg == "--seed" && i + 1 < argc) holdout_seed = std::stoi(argv[++i]);
        else if (arg == "--outlier-th1" && i + 1 < argc) outlier_th1 = std::stof(argv[++i]);
        else if (arg == "--outlier-th2" && i + 1 < argc) outlier_th2 = std::stof(argv[++i]);
    }

    // Discover camera names from calibration folder
    std::vector<std::string> camera_names;
    for (const auto &entry : fs::directory_iterator(calib_folder)) {
        if (entry.path().extension() != ".yaml") continue;
        std::string fname = entry.path().stem().string();
        if (fname.substr(0, 3) == "Cam")
            camera_names.push_back(fname.substr(3));
    }
    std::sort(camera_names.begin(), camera_names.end());
    printf("Found %d cameras\n", (int)camera_names.size());

    FeatureRefinement::FeatureConfig config;
    config.landmarks_file = landmarks_file;
    config.calibration_folder = calib_folder;
    config.output_folder = "/tmp/holdout_compare";
    config.camera_names = camera_names;
    config.prior_rot_weight = rot_prior;
    config.prior_trans_weight = trans_prior;
    config.lock_intrinsics = true;
    config.lock_distortion = true;
    config.ba_outlier_th1 = outlier_th1;
    config.ba_outlier_th2 = outlier_th2;
    config.ba_max_rounds = 5;
    config.ba_convergence_eps = 0.001;
    config.holdout_fraction = 0.2;
    config.holdout_seed = holdout_seed;

    std::string status;
    auto result = FeatureRefinement::run_feature_refinement(config, &status);

    printf("\n========================================\n");
    printf("  HOLDOUT CROSS-VALIDATION RESULTS\n");
    printf("========================================\n\n");
    printf("Landmarks:       %s\n", landmarks_file.c_str());
    printf("Success:         %s\n", result.success ? "YES" : "NO");
    if (!result.success) { printf("Error: %s\n", result.error.c_str()); return 1; }
    printf("Total tracks:    %d\n", result.total_tracks);
    printf("Valid 3D points: %d\n", result.valid_3d_points);
    printf("Outliers:        %d\n", result.ba_outliers_removed);
    printf("Reproj before:   %.3f px\n", result.mean_reproj_before);
    printf("Train reproj:    %.3f px (%d obs)\n", result.train_reproj, result.train_observations);
    printf("Holdout reproj:  %.3f px (%d obs)\n", result.holdout_reproj, result.holdout_observations);
    printf("Holdout/Train:   %.2fx %s\n", result.holdout_ratio,
           result.holdout_ratio > 1.5 ? "*** OVERFITTING ***" : "(OK)");
    printf("BA rounds:       %d\n", result.ba_rounds_completed);
    printf("========================================\n");

    return 0;
}
