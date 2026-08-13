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
        std::cerr << "Usage: " << argv[0] << " <config.json> [--output DIR]"
                     " [--videos FOLDER] [--frame-step N] [--no-sync]\n"
                     "  --videos: run the video path (Cam<serial>.mp4 in FOLDER)\n"
                     "  --no-sync: disable the per-frame timestamp remap\n";
        return 1;
    }

    std::string config_path = argv[1];
    std::string output_dir, videos_folder;
    int frame_step = 1;
    bool use_sync = true;
    for (int i = 2; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--output" && i + 1 < argc) output_dir = argv[++i];
        else if (a == "--videos" && i + 1 < argc) videos_folder = argv[++i];
        else if (a == "--frame-step" && i + 1 < argc) frame_step = std::atoi(argv[++i]);
        else if (a == "--no-sync") use_sync = false;
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

    // Video-path setup: per-frame slot remap when sidecars allow it, else
    // index pairing (the pipeline falls back on an empty map).
    CalibrationPipeline::VideoFrameRange vfr;
    bool is_video = !videos_folder.empty();
    if (is_video) {
        vfr.video_folder = videos_folder;
        vfr.cam_ordered = config.cam_ordered;
        vfr.frame_step = std::max(1, frame_step);
        if (use_sync) {
            std::string sync_status;
            vfr.cam_slot_to_frame = CalibrationPipeline::build_calibration_slot_maps(
                videos_folder, config.cam_ordered, "cam{cam}_timestamps_*.csv",
                &sync_status);
            std::cout << "Sync: " << sync_status << "\n";
        }
    }

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
        config, output_dir, &status, is_video ? &vfr : nullptr, gpu_fn, gpu_ctx);

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
