// run_experimental_calib.cpp — Run experimental calibration pipeline from CLI.
// Uses the same entry point as the RED UI "Experimental" button.

#define STB_IMAGE_IMPLEMENTATION
#include "calibration_pipeline.h"

#ifdef __APPLE__
#include "aruco_metal.h"
#endif

#include "Logger.h"
simplelogger::Logger *logger =
    simplelogger::LoggerFactory::CreateConsoleLogger();

#include <iostream>

int main(int argc, char **argv) {
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <config.json> [--output DIR]\n";
        return 1;
    }

    std::string config_path = argv[1];
    std::string output_dir;
    for (int i = 2; i < argc; i++) {
        if (std::string(argv[i]) == "--output" && i + 1 < argc)
            output_dir = argv[++i];
    }

    CalibrationTool::CalibConfig config;
    std::string error;
    if (!CalibrationTool::parse_config(config_path, config, error)) {
        std::cerr << "Failed to parse config: " << error << "\n";
        return 1;
    }
    std::cout << "Config: " << config.cam_ordered.size() << " cameras\n";
    std::cout << "Images: " << config.img_path << "\n";

    if (output_dir.empty())
        output_dir = std::filesystem::path(config_path).parent_path().string() +
                     "/aruco_image_experimental";

    // Metal GPU acceleration
    aruco_detect::GpuThresholdFunc gpu_fn = nullptr;
    void *gpu_ctx = nullptr;
#ifdef __APPLE__
    auto aruco_metal = aruco_metal_create();
    if (aruco_metal) {
        gpu_fn = aruco_metal_threshold_batch;
        gpu_ctx = aruco_metal;
        std::cout << "Metal GPU: ENABLED\n";
    }
#endif

    std::string status;
    auto result = CalibrationPipeline::run_experimental_pipeline(
        config, output_dir, &status, nullptr, gpu_fn, gpu_ctx);

#ifdef __APPLE__
    if (aruco_metal) aruco_metal_destroy(aruco_metal);
#endif

    if (!result.success) {
        std::cerr << "FAILED: " << result.error << "\n";
        return 1;
    }

    std::cout << "\n=== Results ===\n";
    std::cout << "Per-board reproj:  " << result.mean_reproj_error << " px\n";
    if (result.global_consistency.computed) {
        std::cout << "Multi-view reproj: " << result.global_consistency.mean_reproj << " px\n";
    }
    std::cout << "Output: " << result.output_folder << "\n";
    std::cout << "Status: " << status << "\n";

    return 0;
}
