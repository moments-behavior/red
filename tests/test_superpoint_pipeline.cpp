// test_superpoint_pipeline.cpp — End-to-end test of the unified SuperPoint
// calibration refinement pipeline.
//
// Build: cmake target "test_superpoint_pipeline"
// Run:   ./test_superpoint_pipeline <video_folder> <calib_folder> [output_folder]

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "superpoint_refinement.h"

// Logger required by FFmpegDemuxer
#include "Logger.h"
simplelogger::Logger *logger =
    simplelogger::LoggerFactory::CreateConsoleLogger();

#include <chrono>
#include <cstdio>

int main(int argc, char *argv[]) {
    if (argc < 3) {
        printf("Usage: %s <video_folder> <calib_folder> [output_folder]\n", argv[0]);
        printf("\n  video_folder:  folder with CamXXXXXXX.mp4 synced videos\n");
        printf("  calib_folder:  folder with CamXXXXXXX.yaml calibration files\n");
        printf("  output_folder: where to write results (default: auto)\n");
        return 1;
    }

    std::string video_folder = argv[1];
    std::string calib_folder = argv[2];
    std::string output_folder = argc > 3 ? argv[3] : "";

    // Discover camera names from YAML files
    namespace fs = std::filesystem;
    std::vector<std::string> camera_names;
    for (const auto &entry : fs::directory_iterator(calib_folder)) {
        std::string name = entry.path().filename().string();
        if (name.size() > 7 && name.substr(0, 3) == "Cam" &&
            name.substr(name.size() - 5) == ".yaml") {
            camera_names.push_back(name.substr(3, name.size() - 8));
        }
    }
    std::sort(camera_names.begin(), camera_names.end());
    printf("Found %d cameras\n", (int)camera_names.size());

    // Pick a reference camera (prefer overhead: 2002490 or 710038)
    std::string ref_camera = camera_names.empty() ? "" : camera_names[0];
    for (const auto &cam : camera_names) {
        if (cam == "2002490" || cam == "710038") {
            ref_camera = cam;
            break;
        }
    }
    printf("Reference camera: Cam%s\n", ref_camera.c_str());

    // Configure
    SuperPointRefinement::SPConfig config;
    config.video_folder = video_folder;
    config.calibration_folder = calib_folder;
    config.output_folder = output_folder;
    config.camera_names = camera_names;
    config.ref_camera = ref_camera;
    config.num_frame_sets = 50;
    config.scan_interval_sec = 2.0f;
    config.min_separation_sec = 5.0f;
    config.model_path = "models/superpoint/superpoint.mlpackage";
    config.prior_rot_weight = 10.0;
    config.prior_trans_weight = 100.0;

    auto progress = std::make_shared<SuperPointRefinement::SPProgress>();
    std::string status;

    printf("\n=== Starting SuperPoint Refinement Pipeline ===\n\n");
    auto t0 = std::chrono::steady_clock::now();

    auto result = SuperPointRefinement::run_superpoint_refinement(
        config, &status, progress);

    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t0).count();

    printf("\n=== Pipeline %s in %.1f seconds ===\n",
           result.success ? "COMPLETED" : "FAILED", elapsed);

    if (!result.success) {
        printf("Error: %s\n", result.error.c_str());
        return 1;
    }

    printf("  Frames selected:  %d\n", result.frames_selected);
    printf("  Total tracks:     %d\n", result.total_tracks);
    printf("  3D points:        %d\n", result.valid_3d_points);
    printf("  Observations:     %d\n", result.total_observations);
    printf("  Outliers:         %d\n", result.ba_outliers_removed);
    printf("  Reproj:           %.3f px -> %.3f px\n",
           result.mean_reproj_before, result.mean_reproj_after);
    printf("  BA rounds:        %d\n", result.ba_rounds_completed);
    if (!result.per_round_reproj.empty()) {
        printf("  Per-round reproj: ");
        for (int i = 0; i < (int)result.per_round_reproj.size(); i++)
            printf("%.3f%s", result.per_round_reproj[i],
                   i < (int)result.per_round_reproj.size() - 1 ? " -> " : "");
        printf(" px\n");
    }
    printf("  Output:           %s\n", result.output_folder.c_str());

    printf("\nPer-camera changes:\n");
    printf("  %-12s %10s %10s\n", "Camera", "Rot(deg)", "Trans(mm)");
    for (const auto &cc : result.camera_changes)
        printf("  %-12s %10.4f %10.4f\n",
               cc.name.c_str(), cc.d_rot_deg, cc.d_trans_mm);

    return 0;
}
