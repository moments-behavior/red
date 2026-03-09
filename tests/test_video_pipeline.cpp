// test_video_pipeline.cpp — Test video-based aruco calibration pipeline.
// Uses gpu_calib_test2 project data (16 cameras, ChArUco videos).
//
// Build: cmake target "test_video_pipeline"
// Run:   ./test_video_pipeline

#define STB_IMAGE_IMPLEMENTATION
#include "calibration_pipeline.h"

#ifdef __APPLE__
#include "aruco_metal.h"
#endif

// Logger required by FFmpegDemuxer (NvCodec logging infrastructure)
#include "Logger.h"
simplelogger::Logger *logger =
    simplelogger::LoggerFactory::CreateConsoleLogger();

#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>

static const char *CONFIG_PATH =
    "/Users/johnsonr/datasets/QuanShare/CalibrationDataset/2026_03_01/config.json";
static const char *VIDEO_FOLDER =
    "/Users/johnsonr/datasets/QuanShare/CalibrationDataset/2026_03_01/videos/"
    "2026_03_01_17_05_11";

static int tests_run = 0;
static int tests_passed = 0;

#define TEST_ASSERT(cond, msg) do { \
    tests_run++; \
    if (!(cond)) { \
        std::cerr << "  FAIL: " << msg << " (" #cond ")\n"; \
        return false; \
    } \
    tests_passed++; \
    std::cout << "  PASS: " << msg << "\n"; \
} while (0)

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: discover_aruco_videos
// ─────────────────────────────────────────────────────────────────────────────
static bool test_discover_videos(const CalibrationTool::CalibConfig &config) {
    std::cout << "\n=== Test: discover_aruco_videos ===\n";

    auto videos = CalibrationTool::discover_aruco_videos(
        VIDEO_FOLDER, config.cam_ordered);

    TEST_ASSERT(!videos.empty(), "Videos discovered");
    TEST_ASSERT((int)videos.size() == (int)config.cam_ordered.size(),
        "All 16 cameras have matching videos ("
        + std::to_string(videos.size()) + "/" +
        std::to_string(config.cam_ordered.size()) + ")");

    // Verify each serial maps to a real file
    for (const auto &serial : config.cam_ordered) {
        auto it = videos.find(serial);
        TEST_ASSERT(it != videos.end(),
            "Video found for camera " + serial);
    }

    // Print discovered paths
    for (const auto &[serial, path] : videos) {
        std::cout << "    " << serial << " -> "
                  << std::filesystem::path(path).filename().string() << "\n";
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: get_video_frame_count
// ─────────────────────────────────────────────────────────────────────────────
static bool test_frame_count(const CalibrationTool::CalibConfig &config) {
    std::cout << "\n=== Test: get_video_frame_count ===\n";

    auto videos = CalibrationTool::discover_aruco_videos(
        VIDEO_FOLDER, config.cam_ordered);
    TEST_ASSERT(!videos.empty(), "Videos available");

    int first_count = 0;
    for (const auto &[serial, path] : videos) {
        int count = CalibrationPipeline::get_video_frame_count(path);
        TEST_ASSERT(count > 0, "Frame count > 0 for " + serial +
            " (" + std::to_string(count) + " frames)");
        if (first_count == 0) first_count = count;
    }

    // Verify all cameras have similar frame counts (within 10%)
    for (const auto &[serial, path] : videos) {
        int count = CalibrationPipeline::get_video_frame_count(path);
        double ratio = (double)count / first_count;
        TEST_ASSERT(ratio > 0.9 && ratio < 1.1,
            "Frame count consistency for " + serial +
            " (" + std::to_string(count) + " vs " +
            std::to_string(first_count) + ")");
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: VideoFrameRange construction and frame list
// ─────────────────────────────────────────────────────────────────────────────
static bool test_frame_range() {
    std::cout << "\n=== Test: VideoFrameRange ===\n";

    // Default: start=0, stop=0 (all), step=1
    {
        CalibrationPipeline::VideoFrameRange vfr;
        vfr.start_frame = 0;
        vfr.stop_frame = 100;
        vfr.frame_step = 10;
        int count = 0;
        for (int f = vfr.start_frame; f < vfr.stop_frame; f += vfr.frame_step)
            count++;
        TEST_ASSERT(count == 10, "10 frames with step=10 in range [0,100)");
    }

    // Non-zero start
    {
        CalibrationPipeline::VideoFrameRange vfr;
        vfr.start_frame = 50;
        vfr.stop_frame = 200;
        vfr.frame_step = 25;
        int count = 0;
        for (int f = vfr.start_frame; f < vfr.stop_frame; f += vfr.frame_step)
            count++;
        TEST_ASSERT(count == 6, "6 frames with step=25 in range [50,200)");
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: detect_and_calibrate_intrinsics_video (small frame range)
// ─────────────────────────────────────────────────────────────────────────────
static bool test_intrinsics_video(
    const CalibrationTool::CalibConfig &config,
    aruco_detect::GpuThresholdFunc gpu_fn, void *gpu_ctx) {

    std::cout << "\n=== Test: detect_and_calibrate_intrinsics_video ===\n";

    CalibrationPipeline::VideoFrameRange vfr;
    vfr.video_folder = VIDEO_FOLDER;
    vfr.cam_ordered = config.cam_ordered;
    vfr.start_frame = 0;
    vfr.stop_frame = 0;     // all frames (~389)
    vfr.frame_step = 5;     // every 5th → ~78 frames per camera

    std::map<std::string, CalibrationPipeline::CameraIntrinsics> intrinsics;
    std::vector<int> frame_numbers;
    std::string status;

    auto t0 = std::chrono::steady_clock::now();
    bool ok = CalibrationPipeline::detect_and_calibrate_intrinsics_video(
        config, vfr, intrinsics, &frame_numbers, &status, gpu_fn, gpu_ctx);
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "  Status: " << status << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(1)
              << elapsed << "s\n";

    TEST_ASSERT(ok, "Detection + intrinsics succeeded");
    TEST_ASSERT((int)intrinsics.size() == (int)config.cam_ordered.size(),
        "Intrinsics for all " + std::to_string(config.cam_ordered.size()) +
        " cameras");

    // Check frame numbers
    TEST_ASSERT(!frame_numbers.empty(), "Frame numbers returned");
    TEST_ASSERT((int)frame_numbers.size() > 20,
        "Reasonable number of frames: " + std::to_string(frame_numbers.size()));

    // Check intrinsic quality
    for (const auto &[serial, intr] : intrinsics) {
        TEST_ASSERT(intr.reproj_error > 0 && intr.reproj_error < 5.0,
            "Intrinsic reproj error for " + serial + " = " +
            std::to_string(intr.reproj_error).substr(0, 5) + " px (< 5.0)");
        TEST_ASSERT(intr.image_width > 0 && intr.image_height > 0,
            "Image dimensions for " + serial + " = " +
            std::to_string(intr.image_width) + "x" +
            std::to_string(intr.image_height));
        TEST_ASSERT(intr.K(0, 0) > 100.0,
            "Focal length plausible for " + serial + " (fx=" +
            std::to_string(intr.K(0, 0)).substr(0, 8) + ")");

        // Should have detected corners in at least some frames
        TEST_ASSERT(!intr.corners_per_image.empty(),
            "Corners detected for " + serial + " in " +
            std::to_string(intr.corners_per_image.size()) + " frames");
    }

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5: Full pipeline (video path) — end-to-end with reproj validation
// ─────────────────────────────────────────────────────────────────────────────
static bool test_full_pipeline_video(
    const CalibrationTool::CalibConfig &config,
    aruco_detect::GpuThresholdFunc gpu_fn, void *gpu_ctx) {

    std::cout << "\n=== Test: full pipeline (video) ===\n";

    CalibrationPipeline::VideoFrameRange vfr;
    vfr.video_folder = VIDEO_FOLDER;
    vfr.cam_ordered = config.cam_ordered;
    vfr.start_frame = 0;
    vfr.stop_frame = 0;     // all frames
    vfr.frame_step = 5;     // every 5th frame (some cameras only see board briefly)

    std::string status;
    auto t0 = std::chrono::steady_clock::now();
    auto result = CalibrationPipeline::run_full_pipeline(
        config, "/tmp/calib_video_test_output", &status,
        &vfr, gpu_fn, gpu_ctx);
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "  Status: " << status << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(1)
              << elapsed << "s\n";

    TEST_ASSERT(result.success, "Pipeline succeeded");
    TEST_ASSERT((int)result.cameras.size() == (int)config.cam_ordered.size(),
        "All " + std::to_string(config.cam_ordered.size()) + " cameras calibrated");
    TEST_ASSERT(!result.points_3d.empty(),
        "3D points recovered (" + std::to_string(result.points_3d.size()) + ")");
    TEST_ASSERT(result.image_width > 0 && result.image_height > 0,
        "Image dimensions: " + std::to_string(result.image_width) + "x" +
        std::to_string(result.image_height));

    // Reproj error should be reasonable (< 2.0 px)
    double reproj_threshold = 2.0;
    TEST_ASSERT(result.mean_reproj_error > 0,
        "Reproj error > 0 (" +
        std::to_string(result.mean_reproj_error).substr(0, 6) + " px)");
    TEST_ASSERT(result.mean_reproj_error < reproj_threshold,
        "Reproj error < " + std::to_string(reproj_threshold) + " px (" +
        std::to_string(result.mean_reproj_error).substr(0, 6) + " px)");

    // Verify output files exist
    TEST_ASSERT(!result.output_folder.empty(), "Output folder set");
    {
        namespace fs = std::filesystem;
        TEST_ASSERT(fs::is_directory(result.output_folder),
            "Output folder exists: " + result.output_folder);

        // Check each camera YAML exists
        for (const auto &serial : config.cam_ordered) {
            std::string yaml = result.output_folder + "/Cam" + serial + ".yaml";
            TEST_ASSERT(fs::exists(yaml),
                "YAML exists: Cam" + serial + ".yaml");
        }

        // Check summary data
        TEST_ASSERT(fs::exists(result.output_folder + "/summary_data/intrinsics.json"),
            "Summary intrinsics.json exists");
        TEST_ASSERT(fs::exists(result.output_folder + "/summary_data/landmarks.json"),
            "Summary landmarks.json exists");
    }

    std::cout << "\n  === Pipeline Results ===\n"
              << "  Mean reproj error: " << result.mean_reproj_error << " px\n"
              << "  Cameras: " << result.cameras.size() << "\n"
              << "  3D points: " << result.points_3d.size() << "\n"
              << "  Output: " << result.output_folder << "\n";

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 5b: Full pipeline with step=4 (stress test for sparse camera pairs)
// ─────────────────────────────────────────────────────────────────────────────
static bool test_full_pipeline_step4(
    const CalibrationTool::CalibConfig &config,
    aruco_detect::GpuThresholdFunc gpu_fn, void *gpu_ctx) {

    std::cout << "\n=== Test: full pipeline (video, step=4) ===\n";

    CalibrationPipeline::VideoFrameRange vfr;
    vfr.video_folder = VIDEO_FOLDER;
    vfr.cam_ordered = config.cam_ordered;
    vfr.start_frame = 0;
    vfr.stop_frame = 0;
    vfr.frame_step = 4;

    std::string status;
    auto t0 = std::chrono::steady_clock::now();
    auto result = CalibrationPipeline::run_full_pipeline(
        config, "/tmp/calib_video_step4_test", &status,
        &vfr, gpu_fn, gpu_ctx);
    auto t1 = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t1 - t0).count();

    std::cout << "  Status: " << status << "\n";
    std::cout << "  Time: " << std::fixed << std::setprecision(1)
              << elapsed << "s\n";

    // Step=4 fails on this dataset: peripheral camera pairs (2006516/2008665)
    // share only ~1 frame with the board visible, giving ~12 common points.
    // That's below the 20-point minimum for reliable essential matrix estimation.
    TEST_ASSERT(!result.success,
        "Pipeline correctly rejects sparse data at step=4");
    TEST_ASSERT(result.error.find("common points") != std::string::npos,
        "Error identifies insufficient shared views: " + result.error);

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 6: Compare video vs image pipeline results
// ─────────────────────────────────────────────────────────────────────────────
static bool test_video_vs_image_consistency(
    const CalibrationTool::CalibConfig &config,
    aruco_detect::GpuThresholdFunc gpu_fn, void *gpu_ctx) {

    std::cout << "\n=== Test: video vs image consistency ===\n";

    // Run image-based pipeline (the existing reference)
    std::string img_status;
    auto t0 = std::chrono::steady_clock::now();
    auto img_result = CalibrationPipeline::run_full_pipeline(
        config, "/tmp/calib_img_consistency_test", &img_status,
        nullptr, gpu_fn, gpu_ctx);
    auto t1 = std::chrono::steady_clock::now();
    double img_time = std::chrono::duration<double>(t1 - t0).count();

    TEST_ASSERT(img_result.success, "Image pipeline succeeded");
    std::cout << "  Image pipeline: " << img_result.mean_reproj_error
              << " px in " << std::fixed << std::setprecision(1)
              << img_time << "s\n";

    // Run video-based pipeline with comparable settings
    // Use frame_step=10 for good coverage
    CalibrationPipeline::VideoFrameRange vfr;
    vfr.video_folder = VIDEO_FOLDER;
    vfr.cam_ordered = config.cam_ordered;
    vfr.start_frame = 0;
    vfr.stop_frame = 0;   // all
    vfr.frame_step = 10;

    std::string vid_status;
    t0 = std::chrono::steady_clock::now();
    auto vid_result = CalibrationPipeline::run_full_pipeline(
        config, "/tmp/calib_vid_consistency_test", &vid_status,
        &vfr, gpu_fn, gpu_ctx);
    t1 = std::chrono::steady_clock::now();
    double vid_time = std::chrono::duration<double>(t1 - t0).count();

    TEST_ASSERT(vid_result.success, "Video pipeline succeeded");
    std::cout << "  Video pipeline: " << vid_result.mean_reproj_error
              << " px in " << std::fixed << std::setprecision(1)
              << vid_time << "s\n";

    // Both should produce similar-quality results
    // The video pipeline uses different frames, so exact match isn't expected.
    // But both should be in the sub-pixel range.
    TEST_ASSERT(vid_result.mean_reproj_error < 2.0,
        "Video reproj < 2.0 px (" +
        std::to_string(vid_result.mean_reproj_error).substr(0, 6) + ")");
    TEST_ASSERT(img_result.mean_reproj_error < 2.0,
        "Image reproj < 2.0 px (" +
        std::to_string(img_result.mean_reproj_error).substr(0, 6) + ")");

    // Results should be in the same ballpark (within 5x of each other)
    double ratio = vid_result.mean_reproj_error / img_result.mean_reproj_error;
    TEST_ASSERT(ratio > 0.2 && ratio < 5.0,
        "Reproj ratio video/image = " +
        std::to_string(ratio).substr(0, 4) + " (within 0.2-5.0x)");

    // Both should have the same number of cameras
    TEST_ASSERT(vid_result.cameras.size() == img_result.cameras.size(),
        "Same number of cameras: " +
        std::to_string(vid_result.cameras.size()));

    std::cout << "\n  === Comparison ===\n"
              << "  Image: " << img_result.mean_reproj_error << " px, "
              << img_result.points_3d.size() << " points, "
              << img_time << "s\n"
              << "  Video: " << vid_result.mean_reproj_error << " px, "
              << vid_result.points_3d.size() << " points, "
              << vid_time << "s\n";

    return true;
}

// ─────────────────────────────────────────────────────────────────────────────

int main(int argc, char **argv) {
    // Parse command line: --quick (fast tests only) or --full (all tests)
    bool run_full = false;
    bool run_compare = false;
    for (int i = 1; i < argc; i++) {
        if (std::string(argv[i]) == "--full") run_full = true;
        if (std::string(argv[i]) == "--compare") run_compare = true;
    }

    // Load config
    CalibrationTool::CalibConfig config;
    std::string error;
    if (!CalibrationTool::parse_config(CONFIG_PATH, config, error)) {
        std::cerr << "Failed to parse config: " << error << "\n";
        return 1;
    }
    std::cout << "Config loaded: " << config.cam_ordered.size() << " cameras\n";

    // Create Metal GPU context
    aruco_detect::GpuThresholdFunc gpu_fn = nullptr;
    void *gpu_ctx = nullptr;
#ifdef __APPLE__
    auto aruco_metal = aruco_metal_create();
    if (aruco_metal) {
        gpu_fn = aruco_metal_threshold_batch;
        gpu_ctx = aruco_metal;
        std::cout << "Metal GPU acceleration: ENABLED\n";
    } else {
        std::cout << "Metal GPU acceleration: FAILED (using CPU)\n";
    }
#endif

    // Run tests
    bool all_pass = true;

    // Fast tests (always run)
    all_pass &= test_discover_videos(config);
    all_pass &= test_frame_count(config);
    all_pass &= test_frame_range();
    all_pass &= test_intrinsics_video(config, gpu_fn, gpu_ctx);

    // Full pipeline tests (--full flag)
    if (run_full) {
        all_pass &= test_full_pipeline_video(config, gpu_fn, gpu_ctx);
        all_pass &= test_full_pipeline_step4(config, gpu_fn, gpu_ctx);
    } else {
        std::cout << "\n(Skipping full pipeline tests — use --full to run)\n";
    }

    // Comparison test (--compare flag, takes ~2-3 min)
    if (run_compare) {
        all_pass &= test_video_vs_image_consistency(config, gpu_fn, gpu_ctx);
    } else {
        std::cout << "(Skipping comparison test — use --compare to run)\n";
    }

#ifdef __APPLE__
    aruco_metal_destroy(aruco_metal);
#endif

    // Summary
    std::cout << "\n════════════════════════════════════\n";
    std::cout << "Tests: " << tests_passed << "/" << tests_run << " passed\n";
    if (all_pass) {
        std::cout << "RESULT: ALL PASSED\n";
    } else {
        std::cout << "RESULT: SOME FAILED\n";
    }
    std::cout << "════════════════════════════════════\n";

    return all_pass ? 0 : 1;
}
