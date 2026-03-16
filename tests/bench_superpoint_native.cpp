// bench_superpoint_native.cpp — End-to-end benchmark of the native SuperPoint pipeline.
// Runs the full pipeline (VT decode → CoreML → BLAS match → track build → BA)
// on a real multi-camera dataset with timing breakdowns.

#include "superpoint_refinement.h"
#include <chrono>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

namespace fs = std::filesystem;

int main(int argc, char **argv) {
    // Default paths for the rat 17-camera dataset
    std::string video_folder = "/Users/johnsonr/datasets/rat/sessions/2025_09_03_15_18_21";
    std::string calib_folder = "/Users/johnsonr/datasets/rat/sessions/2025_09_03_15_18_21/2025_08_14_09_23_31_results/calibration";
    std::string model_path = "models/superpoint/superpoint.mlpackage";
    std::string output_folder = "/tmp/superpoint_bench";
    int num_frames = 50;
    float ratio_threshold = 0.8f;
    float rot_prior = 10.0f;
    float trans_prior = 100.0f;
    float reproj_thresh = 15.0f;
    int max_keypoints = 4096;

    // Parse optional args
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--videos" && i + 1 < argc) video_folder = argv[++i];
        else if (arg == "--calib" && i + 1 < argc) calib_folder = argv[++i];
        else if (arg == "--model" && i + 1 < argc) model_path = argv[++i];
        else if (arg == "--output" && i + 1 < argc) output_folder = argv[++i];
        else if (arg == "--frames" && i + 1 < argc) num_frames = std::stoi(argv[++i]);
        else if (arg == "--ratio" && i + 1 < argc) ratio_threshold = std::stof(argv[++i]);
        else if (arg == "--rot-prior" && i + 1 < argc) rot_prior = std::stof(argv[++i]);
        else if (arg == "--trans-prior" && i + 1 < argc) trans_prior = std::stof(argv[++i]);
        else if (arg == "--reproj-thresh" && i + 1 < argc) reproj_thresh = std::stof(argv[++i]);
        else if (arg == "--max-keypoints" && i + 1 < argc) max_keypoints = std::stoi(argv[++i]);
        else {
            printf("Usage: %s [--frames N] [--ratio F] [--rot-prior F] [--trans-prior F] [--reproj-thresh F] [--max-keypoints N]\n", argv[0]);
            return 1;
        }
    }

    // Discover cameras: find .mp4 files that have matching .yaml calibration
    printf("=== SuperPoint Native Pipeline Benchmark ===\n\n");
    printf("Video folder:  %s\n", video_folder.c_str());
    printf("Calib folder:  %s\n", calib_folder.c_str());
    printf("Model path:    %s\n", model_path.c_str());
    printf("Output folder: %s\n", output_folder.c_str());
    printf("Num frames:    %d\n", num_frames);
    printf("Ratio thresh:  %.2f\n", ratio_threshold);
    printf("Rot prior:     %.1f\n", rot_prior);
    printf("Trans prior:   %.1f\n", trans_prior);
    printf("Reproj thresh: %.1f px\n", reproj_thresh);
    printf("Max keypoints: %d\n\n", max_keypoints);

    if (!fs::exists(video_folder)) { printf("ERROR: Video folder not found\n"); return 1; }
    if (!fs::exists(calib_folder)) { printf("ERROR: Calib folder not found\n"); return 1; }
    if (!fs::exists(model_path)) { printf("ERROR: Model not found: %s\n", model_path.c_str()); return 1; }

    std::vector<std::string> camera_names;
    for (const auto &entry : fs::directory_iterator(video_folder)) {
        if (entry.path().extension() != ".mp4") continue;
        std::string fname = entry.path().stem().string();
        if (fname.substr(0, 3) != "Cam") continue;
        std::string serial = fname.substr(3);
        // Check matching calibration YAML
        std::string yaml_path = calib_folder + "/Cam" + serial + ".yaml";
        if (fs::exists(yaml_path))
            camera_names.push_back(serial);
    }
    std::sort(camera_names.begin(), camera_names.end());

    printf("Found %d cameras with calibration:\n", (int)camera_names.size());
    for (const auto &name : camera_names)
        printf("  Cam%s\n", name.c_str());
    printf("\n");

    // Configure pipeline
    SuperPointRefinement::SPConfig config;
    config.video_folder = video_folder;
    config.calibration_folder = calib_folder;
    config.output_folder = output_folder;
    config.camera_names = camera_names;
    config.ref_camera = camera_names[0];
    config.num_frame_sets = num_frames;
    config.min_separation_sec = 5.0f;
    config.model_path = model_path;
    config.max_keypoints = max_keypoints;
    config.ratio_threshold = ratio_threshold;
    config.reproj_thresh = reproj_thresh;
    config.prior_rot_weight = rot_prior;
    config.prior_trans_weight = trans_prior;
    config.lock_intrinsics = true;
    config.lock_distortion = true;
    config.ba_outlier_th1 = 10.0;
    config.ba_outlier_th2 = 3.0;
    config.ba_max_rounds = 5;
    config.ba_convergence_eps = 0.001;

    auto progress = std::make_shared<SuperPointRefinement::SPProgress>();
    std::string status;

    // Run the full pipeline
    printf("Starting pipeline...\n\n");
    auto t0 = std::chrono::steady_clock::now();

    auto result = SuperPointRefinement::run_superpoint_refinement(config, &status, progress);

    auto elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();

    // Report results
    printf("\n");
    printf("========================================\n");
    printf("  BENCHMARK RESULTS\n");
    printf("========================================\n\n");

    if (result.success) {
        printf("Status:          SUCCESS\n");
        printf("Total time:      %.1f seconds\n", elapsed);
        printf("Frames selected: %d\n", result.frames_selected);
        printf("Total tracks:    %d\n", result.total_tracks);
        printf("Valid 3D points: %d\n", result.valid_3d_points);
        printf("Total obs:       %d\n", result.total_observations);
        printf("Outliers:        %d\n", result.ba_outliers_removed);
        printf("Reproj before:   %.3f px\n", result.mean_reproj_before);
        printf("Reproj after:    %.3f px (train)\n", result.mean_reproj_after);
        if (result.holdout_observations > 0) {
            printf("Holdout reproj:  %.3f px (%d obs)\n", result.holdout_reproj, result.holdout_observations);
            printf("Holdout/Train:   %.2fx %s\n", result.holdout_ratio,
                   result.holdout_ratio > 1.5 ? "*** OVERFITTING ***" : "(OK)");
        }
        printf("BA rounds:       %d\n", result.ba_rounds_completed);
        printf("Output:          %s\n", result.output_folder.c_str());

        if (!result.per_round_reproj.empty()) {
            printf("\nPer-round reproj: ");
            for (int i = 0; i < (int)result.per_round_reproj.size(); i++)
                printf("%.3f%s", result.per_round_reproj[i],
                       i < (int)result.per_round_reproj.size() - 1 ? " → " : "");
            printf(" px\n");
        }

        if (!result.camera_changes.empty()) {
            printf("\n%-12s %8s %10s\n", "Camera", "Rot(deg)", "Trans(mm)");
            printf("%-12s %8s %10s\n", "------", "--------", "--------");
            for (const auto &ch : result.camera_changes)
                printf("%-12s %8.4f %10.4f\n", ch.name.c_str(), ch.d_rot_deg, ch.d_trans_mm);
        }

        printf("\n  %.1f seconds for %d cameras x %d frames\n", elapsed,
               (int)camera_names.size(), num_frames);
        printf("  = %.1f ms per camera-frame (VT decode + CoreML + match)\n",
               elapsed * 1000.0 / (camera_names.size() * num_frames));
    } else {
        printf("Status:  FAILED\n");
        printf("Error:   %s\n", result.error.c_str());
        printf("Status:  %s\n", status.c_str());
    }

    printf("\n========================================\n\n");
    return result.success ? 0 : 1;
}
