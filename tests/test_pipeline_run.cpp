// test_pipeline_run.cpp — Run full calibration pipeline and report results.
// Build: cmake target "test_pipeline_run"
// Run:   ./test_pipeline_run [--experimental] [--video] [--step N]

#define STB_IMAGE_IMPLEMENTATION
#include "calibration_pipeline.h"

#ifdef __APPLE__
#include "aruco_metal.h"
#endif

// Logger required by FFmpegDemuxer (NvCodec logging infrastructure)
#include "Logger.h"
simplelogger::Logger *logger =
    simplelogger::LoggerFactory::CreateConsoleLogger();

#include <iostream>

static const char *CONFIG_PATH =
    "/Users/johnsonr/datasets/QuanShare/CalibrationDataset/2026_03_01/config.json";
static const char *VIDEO_FOLDER =
    "/Users/johnsonr/datasets/QuanShare/CalibrationDataset/2026_03_01/videos/"
    "2026_03_01_17_05_11";

int main(int argc, char **argv) {
    bool run_experimental = false;
    bool run_video = false;
    int frame_step = 1;
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--experimental") run_experimental = true;
        else if (arg == "--video") run_video = true;
        else if (arg == "--step" && i + 1 < argc) frame_step = std::atoi(argv[++i]);
    }

    CalibrationTool::CalibConfig config;
    std::string error;
    if (!CalibrationTool::parse_config(CONFIG_PATH, config, error)) {
        std::cerr << "Failed to parse config: " << error << "\n";
        return 1;
    }
    std::cout << "Config loaded: " << config.cam_ordered.size() << " cameras\n";
    if (run_video)
        std::cout << "Mode: VIDEO (step=" << frame_step << ")\n";
    else
        std::cout << "Mode: IMAGES\n";

    // Create Metal GPU context for accelerated threshold (macOS only)
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
#else
    std::cout << "Metal GPU acceleration: N/A (Linux)\n";
#endif

    // Set up video frame range if needed
    CalibrationPipeline::VideoFrameRange vfr;
    CalibrationPipeline::VideoFrameRange *vfr_ptr = nullptr;
    if (run_video) {
        vfr.video_folder = VIDEO_FOLDER;
        vfr.cam_ordered = config.cam_ordered;
        vfr.start_frame = 0;
        vfr.stop_frame = 0; // all frames
        vfr.frame_step = frame_step;
        vfr_ptr = &vfr;
    }

    std::string status;
    CalibrationPipeline::CalibrationResult result;

    std::string out_base = run_video
        ? "/tmp/calib_test_video_step" + std::to_string(frame_step)
        : "/tmp/calib_test_output";

    if (run_experimental) {
        std::cout << "\n=== Running EXPERIMENTAL pipeline ===\n\n";
        result = CalibrationPipeline::run_experimental_pipeline(
            config, out_base + "_experimental", &status,
            vfr_ptr, gpu_fn, gpu_ctx);
    } else {
        result = CalibrationPipeline::run_full_pipeline(
            config, out_base, &status,
            vfr_ptr, gpu_fn, gpu_ctx);
    }

#ifdef __APPLE__
    aruco_metal_destroy(aruco_metal);
#endif

    std::cout << "Status: " << status << "\n";
    if (!result.success) {
        std::cerr << "Pipeline failed: " << result.error << "\n";
        return 1;
    }

    std::cout << "\nMean reproj error: " << result.mean_reproj_error << " px\n";
    if (!result.all_reproj_errors.empty()) {
        auto sorted = result.all_reproj_errors;
        std::sort(sorted.begin(), sorted.end());
        std::cout << "Median reproj error: " << sorted[sorted.size()/2] << " px\n";
    }
    std::cout << "Cameras: " << result.cameras.size() << "\n";
    std::cout << "3D points: " << result.points_3d.size() << "\n";
    std::cout << "Output: " << result.output_folder << "\n";

    if (run_experimental && !result.per_camera_metrics.empty()) {
        std::cout << "\nPer-camera metrics:\n";
        for (const auto &m : result.per_camera_metrics) {
            std::cout << "  " << m.name
                      << ": dets=" << m.detection_count
                      << " obs=" << m.observation_count
                      << " mean=" << m.mean_reproj
                      << " median=" << m.median_reproj
                      << " std=" << m.std_reproj
                      << " max=" << m.max_reproj << "\n";
        }
        std::cout << "BA rounds: " << result.ba_rounds
                  << "  Outliers removed: " << result.outliers_removed << "\n";
    }

    // Validation
    double threshold = 15.0;
    if (result.mean_reproj_error > threshold) {
        std::cerr << "\nFAILED: mean reproj error " << result.mean_reproj_error
                  << " > " << threshold << " px threshold\n";
        return 1;
    }
    std::cout << "\nPASSED: mean reproj error within threshold\n";
    return 0;
}
