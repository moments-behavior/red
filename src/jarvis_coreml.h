#pragma once
// jarvis_coreml.h — JARVIS pose estimation via native CoreML
//
// Loads .mlpackage/.mlmodelc models and runs CenterDetect + KeypointDetect
// directly on CVPixelBuffers from the VideoToolbox decode pipeline.
// Zero-copy input: CoreML reads IOSurface-backed CVPixelBuffers on GPU/ANE.
//
// macOS only. Preferred over ONNX Runtime on Apple Silicon (~6-20ms/frame).

#ifdef __APPLE__

#include "annotation.h"
#include "types.h"
#include <string>
#include <vector>
#include <chrono>
#include <CoreVideo/CoreVideo.h>

#include "jarvis_model_config.h"

// Forward declarations to avoid heavy includes
struct SkeletonContext;

struct JarvisCoreMLState {
    bool loaded = false;
    bool available = false; // set true if macOS 13+
    std::string status;

    // Opaque pointers to MLModel instances
    void *center_model = nullptr;
    void *keypoint_model = nullptr;

    // Full model config (shared struct with ONNX backend)
    JarvisModelConfig config;

    // Flat copies for hot-path inference access (avoid indirection)
    int center_input_size = 320;
    int keypoint_input_size = 704;
    int num_joints = 24;

    // Timing (per jarvis_coreml_predict_frame call)
    float last_center_ms = 0;
    float last_keypoint_ms = 0;
    float last_total_ms = 0;
};

// Initialize: load .mlpackage (compiles to .mlmodelc on first use, cached after).
// model_dir: directory containing center_detect.mlpackage/ and keypoint_detect.mlpackage/
// Pass pre-parsed config to avoid redundant JSON I/O.
bool jarvis_coreml_init(JarvisCoreMLState &s, const std::string &model_dir,
                         const JarvisModelConfig &cfg);

// Check if CoreML is available on this system
bool jarvis_coreml_available();

// Run full prediction on one frame across all cameras.
// pixel_buffers: CVPixelBufferRef array, one per camera (BGRA, IOSurface-backed)
bool jarvis_coreml_predict_frame(
    JarvisCoreMLState &s,
    AnnotationMap &amap, u32 frame_num,
    const std::vector<CVPixelBufferRef> &pixel_buffers,
    const std::vector<int> &cam_widths,
    const std::vector<int> &cam_heights,
    const SkeletonContext &skeleton,
    int num_cameras,
    float confidence_threshold = 0.1f);

void jarvis_coreml_cleanup(JarvisCoreMLState &s);

#endif // __APPLE__
