// test_calib_reload.cpp — Run experimental calibration, then reload from folder
// and compare all fields side-by-side.
// Build: cmake target "test_calib_reload"
// Run:   ./test_calib_reload

#define STB_IMAGE_IMPLEMENTATION
#include "calibration_pipeline.h"
#include "calibration_tool.h"

#ifdef __APPLE__
#include "aruco_metal.h"
#endif

// Logger required by FFmpegDemuxer (NvCodec logging infrastructure)
#include "Logger.h"
simplelogger::Logger *logger =
    simplelogger::LoggerFactory::CreateConsoleLogger();

#include <cmath>
#include <cstdio>

static const char *CONFIG_PATH =
    "/Users/johnsonr/datasets/rat/calibration_images/original_calibration_results/config.json";

int main() {
    // 1. Parse config
    CalibrationTool::CalibConfig config;
    std::string err;
    if (!CalibrationTool::parse_config(CONFIG_PATH, config, err)) {
        printf("Config parse error: %s\n", err.c_str());
        return 1;
    }
    printf("Config loaded: %d cameras, dictionary=%d\n",
           (int)config.cam_ordered.size(), config.charuco_setup.dictionary);

    // 2. Run experimental pipeline
    std::string status;
    std::string base = "/tmp/red_calib_test";

#ifdef __APPLE__
    auto am = aruco_metal_create();
    aruco_detect::GpuThresholdFunc gfn = am ? aruco_metal_threshold_batch : nullptr;
    auto result = CalibrationPipeline::run_experimental_pipeline(
        config, base, &status, nullptr, gfn, am);
    if (am) aruco_metal_destroy(am);
#else
    auto result = CalibrationPipeline::run_experimental_pipeline(
        config, base, &status);
#endif

    printf("\n=== INITIAL RUN ===\n");
    printf("Success: %d\n", result.success);
    printf("Error: %s\n", result.error.c_str());
    printf("Warning: %s\n", result.warning.c_str());
    printf("Cameras: %d\n", (int)result.cameras.size());
    printf("cam_names: %d\n", (int)result.cam_names.size());
    printf("Mean reproj: %.6f px\n", result.mean_reproj_error);
    printf("Output: %s\n", result.output_folder.c_str());

    if (!result.success) {
        printf("Pipeline failed!\n");
        return 1;
    }

    printf("\nPer-camera metrics (initial):\n");
    printf("%-12s %6s %8s %8s %8s\n", "Camera", "Dets", "Mean", "Median", "Max");
    for (const auto &m : result.per_camera_metrics) {
        printf("%-12s %6d %8.4f %8.4f %8.4f\n",
               m.name.c_str(), m.detection_count, m.mean_reproj, m.median_reproj, m.max_reproj);
    }

    // 3. Reload from folder
    auto reload = CalibrationPipeline::load_calibration_from_folder(
        result.output_folder, config.cam_ordered);

    printf("\n=== RELOAD ===\n");
    printf("Success: %d\n", reload.success);
    printf("Error: %s\n", reload.error.c_str());
    printf("Warning: %s\n", reload.warning.c_str());
    printf("Cameras: %d\n", (int)reload.cameras.size());
    printf("cam_names: %d\n", (int)reload.cam_names.size());
    printf("Mean reproj: %.6f px\n", reload.mean_reproj_error);

    printf("\nPer-camera metrics (reload):\n");
    printf("%-12s %6s %8s %8s %8s\n", "Camera", "Dets", "Mean", "Median", "Max");
    for (const auto &m : reload.per_camera_metrics) {
        printf("%-12s %6d %8.4f %8.4f %8.4f\n",
               m.name.c_str(), m.detection_count, m.mean_reproj, m.median_reproj, m.max_reproj);
    }

    // 4. Compare
    printf("\n=== COMPARISON ===\n");
    printf("cam_names match: %s\n",
           (result.cam_names == reload.cam_names) ? "YES" : "NO");
    printf("cameras count: %d vs %d\n",
           (int)result.cameras.size(), (int)reload.cameras.size());
    printf("mean_reproj: %.6f vs %.6f (delta=%.6f)\n",
           result.mean_reproj_error, reload.mean_reproj_error,
           std::abs(result.mean_reproj_error - reload.mean_reproj_error));

    // Per-camera comparison
    printf("\n%-12s %10s %10s %10s\n", "Camera", "Init Mean", "Load Mean", "Delta");
    int n = std::min(result.per_camera_metrics.size(), reload.per_camera_metrics.size());
    double max_delta = 0;
    for (int i = 0; i < n; i++) {
        auto &a = result.per_camera_metrics[i];
        auto &b = reload.per_camera_metrics[i];
        double d = std::abs(a.mean_reproj - b.mean_reproj);
        if (d > max_delta) max_delta = d;
        printf("%-12s %10.6f %10.6f %10.6f %s\n",
               a.name.c_str(), a.mean_reproj, b.mean_reproj, d,
               d > 0.001 ? "*** MISMATCH" : "OK");
    }
    printf("\nMax per-camera delta: %.6f px\n", max_delta);
    printf("Overall: %s\n", max_delta < 0.001 ? "PASS" : "FAIL");

    // =========================================================
    // 5. Simulate Flip Z and verify roundtrip
    // =========================================================
    printf("\n=== FLIP Z ROUNDTRIP ===\n");

    // Apply Flip Z to a copy of the result
    auto flipped = result;
    Eigen::Matrix3d F = Eigen::Vector3d(1, 1, -1).asDiagonal();
    for (auto &cam : flipped.cameras)
        cam.R = cam.R * F;
    for (auto &[id, pt] : flipped.points_3d)
        pt.z() = -pt.z();

    // Verify det(R) = -1
    printf("det(R[0]) after flip: %.1f (should be -1)\n",
           flipped.cameras[0].R.determinant());

    // Save flipped data
    {
        std::string werr;
        CalibrationPipeline::write_calibration(
            flipped.cameras, flipped.cam_names, flipped.output_folder,
            flipped.image_width, flipped.image_height, &werr);
        namespace fs = std::filesystem;
        std::string pts_path = flipped.output_folder +
            "/summary_data/bundle_adjustment/ba_points.json";
        if (fs::exists(pts_path)) {
            nlohmann::json pts_j;
            for (const auto &[id, pt] : flipped.points_3d)
                pts_j[std::to_string(id)] = {pt.x(), pt.y(), pt.z()};
            std::ofstream pf(pts_path);
            pf << pts_j.dump(2);
        }
        printf("Saved flipped YAMLs + ba_points.json\n");
    }

    // Reload flipped data
    auto reload_flip = CalibrationPipeline::load_calibration_from_folder(
        flipped.output_folder, config.cam_ordered);

    printf("Reload after flip: success=%d, cameras=%d, mean_reproj=%.6f px\n",
           reload_flip.success, (int)reload_flip.cameras.size(),
           reload_flip.mean_reproj_error);

    // Compare original vs reload-after-flip
    printf("\n%-12s %10s %10s %10s\n", "Camera", "Original", "FlipReload", "Delta");
    double max_flip_delta = 0;
    for (const auto &a : result.per_camera_metrics) {
        double b_mean = -1;
        for (const auto &m : reload_flip.per_camera_metrics)
            if (m.name == a.name) { b_mean = m.mean_reproj; break; }
        if (b_mean < 0) continue;
        double d = std::abs(a.mean_reproj - b_mean);
        if (d > max_flip_delta) max_flip_delta = d;
        printf("%-12s %10.6f %10.6f %10.6f %s\n",
               a.name.c_str(), a.mean_reproj, b_mean, d,
               d > 0.01 ? "*** MISMATCH" : "OK");
    }
    printf("\nMax delta: %.6f px\n", max_flip_delta);
    printf("mean_reproj: %.6f vs %.6f (delta=%.6f)\n",
           result.mean_reproj_error, reload_flip.mean_reproj_error,
           std::abs(result.mean_reproj_error - reload_flip.mean_reproj_error));
    printf("Flip Z roundtrip: %s\n", max_flip_delta < 0.01 ? "PASS" : "FAIL");

    return 0;
}
